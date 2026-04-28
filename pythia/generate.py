from __future__ import annotations

import re
import textwrap

from ollama import ChatResponse, chat

from pythia import settings

SYSTEM_PROMPT = textwrap.dedent(
    """\
    You are a technical assistant that answers questions about electronic
    component datasheets. You will be given retrieved context passages from
    one or more DC-DC converter datasheets.

    Rules:
    1. Answer ONLY from the provided context. If the context does not contain
       enough information, say so explicitly — do not guess or hallucinate.
    2. When citing a specific value (voltage, current, frequency, etc.),
       include the unit and any qualifying conditions (e.g. temperature, input
       voltage, load current).
    3. If the answer involves a range (min/typ/max), state all available
       bounds.
    4. When the context contains a table or enumerated list of thresholds
       relevant to the question, include all rows/entries rather than
       summarizing with a single example.
    5. If a parameter behaves differently under specific conditions (e.g.,
       fault, light load, startup), explain what triggers the change and
       how the behavior differs from normal operation.
    6. If the context contains apparently contradictory specifications, note
       and reconcile them (e.g., a "fixed" frequency that changes during fault
       conditions).
    7. Cite the specific sections that support your answer using
       [Section x.y.z, Page n] format. If no section number is available, cite
       the page number.
    8. Keep answers focused and avoid restating the question, but prioritize
       completeness of technical detail over brevity.
"""
)


def build_prompt(query: str, context_chunks: list[dict]) -> list[dict]:
    context_block = ""
    for chunk in context_chunks:
        source = chunk.get("component", "unknown")
        text = chunk["text"]
        if chunk["section_number"] is not None:
            context_block += f"[Page {chunk["page"]}, Section {chunk["section_number"]} — {source}]\n{text}\n\n"
        else:
            context_block += f"[Page {chunk["page"]} — {source}]\n{text}\n\n"

    user_content = (
        f"Context:\n{context_block}" f"Question: {query}\n\n" f"Answer the question using only the context above."
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def generate(
    query: str,
    context_chunks: list[dict],
    think: bool = True,
) -> tuple[str, str]:
    messages = build_prompt(query, context_chunks)

    response: ChatResponse = chat(
        model=settings.generation.ollama_model,
        messages=messages,
        options={
            "temperature": 0.2,  # low temp for factual grounding
            "num_ctx": 8192,  # context window for long chunks
            "num_predict": 4096,  # cap output length
        },
        think=think,
    )

    content = response.message.content or ""
    thinking = response.message.thinking or ""

    return content, thinking


def strip_think_tags(text: str) -> str:
    """Remove <think>...</think> blocks if present in raw output."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
