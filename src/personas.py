from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Persona:
    name: str
    system_prompt: str
    style_instructions: str
    max_tokens_hint: int


TECHNICAL = Persona(
    name="technical",
    system_prompt=(
        "You are a technical writer. Produce precise, well-structured summaries "
        "that preserve domain-specific terminology, statistics, and quantitative "
        "details. Use formal language and structured output."
    ),
    style_instructions=(
        "Preserve technical jargon and acronyms. Include all statistics, "
        "percentages, and numerical data. Use structured paragraphs."
    ),
    max_tokens_hint=512,
)

CASUAL = Persona(
    name="casual",
    system_prompt=(
        "You are a friendly writer. Produce clear, conversational summaries "
        "in plain language that anyone can understand. Keep it short and "
        "approachable."
    ),
    style_instructions=(
        "Use plain language. Avoid jargon. Keep sentences short and "
        "conversational. Explain technical terms if they must appear."
    ),
    max_tokens_hint=256,
)

EXECUTIVE = Persona(
    name="executive",
    system_prompt=(
        "You are a business analyst. Produce concise executive briefings "
        "with bullet-point conclusions, key metrics, and action items. "
        "Decision-makers will read this."
    ),
    style_instructions=(
        "Use bullet points. Lead with the conclusion. Highlight key metrics "
        "and action items. No filler, no background context."
    ),
    max_tokens_hint=200,
)

ACADEMIC = Persona(
    name="academic",
    system_prompt=(
        "You are an academic researcher. Produce scholarly summaries that "
        "use formal register, preserve citations and attribution, and note "
        "methodology where relevant."
    ),
    style_instructions=(
        "Use formal academic register. Preserve citations and source "
        "attribution. Note methodology and limitations. Use hedged language "
        "where appropriate (e.g. 'suggests', 'indicates')."
    ),
    max_tokens_hint=512,
)

DEFAULT = CASUAL

PERSONAS: dict[str, Persona] = {
    "technical": TECHNICAL,
    "casual": CASUAL,
    "executive": EXECUTIVE,
    "academic": ACADEMIC,
    "default": DEFAULT,
}

LENGTH_MULTIPLIERS: dict[str, float] = {
    "brief": 0.5,
    "standard": 1.0,
    "detailed": 2.0,
}


def get_persona(name: str) -> Persona:
    """Look up a persona by name. Raises ValueError for unknown names."""
    persona = PERSONAS.get(name.lower())
    if persona is None:
        valid = ", ".join(sorted(PERSONAS.keys()))
        raise ValueError(f"Unknown persona '{name}'. Valid options: {valid}")
    return persona


def format_prompt(
    persona: Persona,
    extracted_sentences: list[str],
    length: str = "standard",
) -> str:
    """Build the user prompt combining persona style, extracted text, and length."""
    multiplier = LENGTH_MULTIPLIERS.get(length, 1.0)
    max_tokens = int(persona.max_tokens_hint * multiplier)

    sentences_block = "\n".join(
        f"- {s}" for s in extracted_sentences if s.strip()
    )

    return (
        f"{persona.style_instructions}\n\n"
        f"Summarize the following extracted sentences in at most "
        f"{max_tokens} tokens:\n\n"
        f"{sentences_block}"
    )
