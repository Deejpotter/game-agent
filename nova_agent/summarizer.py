"""
nova_agent.summarizer
=====================
Context summarization for managing long conversation histories.

Problem
-------
Pokémon playthroughs generate enormous conversation histories.  After
``MAX_HISTORY_MESSAGES`` turns the context window fills up and the model
starts losing track of early events.

Solution (inspired by Anthropic's Claude Plays Pokemon approach)
----------------------------------------------------------------
When history exceeds the threshold:
1. Ask the model to write a dense summary of the last N turns.
2. Insert the summary as a "recap" system message at the top of history.
3. Clear the raw turn-by-turn messages.
4. Ask the model to review and refresh the knowledge-base sections that
   may have become stale (a second cheap call).

This gives the model a compressed but accurate picture of its own history
without losing the KB's structured memory.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from openai import OpenAI

if TYPE_CHECKING:
    from nova_agent.memory import KnowledgeBase


_SUMMARIZE_PROMPT = """\
You are summarizing your own recent gameplay history for a Pokemon game.
Write a concise but detailed summary covering:
- Where you are on the map (use coordinates if known)
- What you were trying to do
- What battles you fought and the outcome
- Important items or story events
- Any problems you encountered (getting stuck, failing to navigate, etc.)
- Your current team's health status

Be factual. Keep it under 400 words.
"""

_KB_REVIEW_PROMPT = """\
Review the knowledge base sections below and identify any that are outdated,
incorrect, or missing important information based on the gameplay summary.

Return a JSON object where each key is a section name and each value is the
improved content for that section. Only include sections that need updating.
Return {{}} if nothing needs changing.

Valid sections: current_status, game_progress, objectives, party_status, notes

Knowledge base:
{kb_block}

Recent summary:
{summary}
"""


def summarize_and_reset(
    client: OpenAI,
    model: str,
    history: list[dict[str, Any]],
    kb: "KnowledgeBase",
    *,
    extra_body: dict | None = None,
) -> list[dict[str, Any]]:
    """Summarize history, reset the message list, and refresh the KB.

    Args:
        client: OpenAI-compatible client.
        model: Model name to use for summarization.
        history: Current conversation history (list of message dicts).
        kb: KnowledgeBase instance to update.
        extra_body: Optional extra body fields (e.g. enable_thinking).

    Returns:
        New history list with just the summary recap message.
    """
    if not history:
        return history

    print(f"[summarizer] Summarizing {len(history)} messages…")

    # ── Step 1: Generate summary ──────────────────────────────────────────
    summary_messages = [
        {"role": "system", "content": _SUMMARIZE_PROMPT},
        *history,
        {"role": "user", "content": "Please write the summary now."},
    ]

    try:
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": summary_messages,
            "max_tokens": 1024,
        }
        if extra_body:
            kwargs["extra_body"] = extra_body

        resp = client.chat.completions.create(**kwargs)
        summary_text = (resp.choices[0].message.content or "").strip()
    except Exception as exc:
        print(f"[summarizer] Summary call failed: {exc}")
        # Don't reset history if we can't summarize.
        return history

    print(f"[summarizer] Summary ({len(summary_text)} chars): {summary_text[:120]}…")

    # ── Step 2: KB review ─────────────────────────────────────────────────
    try:
        review_prompt = _KB_REVIEW_PROMPT.format(
            kb_block=kb.to_prompt_block(),
            summary=summary_text,
        )
        review_messages = [
            {"role": "user", "content": review_prompt},
        ]
        review_kwargs: dict[str, Any] = {
            "model": model,
            "messages": review_messages,
            "max_tokens": 512,
        }
        review_resp = client.chat.completions.create(**review_kwargs)
        review_text = (review_resp.choices[0].message.content or "").strip()

        import json, re
        # Extract JSON from the response — model may wrap it in ```json ... ``` blocks.
        _json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", review_text, re.DOTALL)
        if _json_match:
            review_text = _json_match.group(1)
        else:
            # Try to find a bare JSON object.
            _bare = re.search(r"\{.*\}", review_text, re.DOTALL)
            if _bare:
                review_text = _bare.group(0)
        updates: dict[str, str] = json.loads(review_text)
        for section, content in updates.items():
            result = kb.update_section(section, content)
            print(f"[summarizer] KB review → {result}")

    except Exception as exc:
        print(f"[summarizer] KB review failed (non-fatal): {exc}")

    # ── Step 3: Reset history with summary as first message ───────────────
    recap_message: dict[str, Any] = {
        "role": "assistant",
        "content": f"[GAME PROGRESS RECAP]\n{summary_text}",
    }
    print("[summarizer] History reset with recap.")
    return [recap_message]
