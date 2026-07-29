"""Heuristic assembled_vs_imported + confidence flag (SPEC §2.1 note, §4).

The field is not clean: it hides in titles/descriptions ("imported", "unregistered",
auction-sheet mentions). Derive it, do not assume a tidy column.

This is the canonical home of the heuristic. Stage 2 of the scrape (which has the live
title/description text) imports it from here and stores the ``(label, confidence)`` pair
onto each listing, so downstream feature code consumes those two columns directly.
"""

from __future__ import annotations

# Signals that a car is imported when no clean "Assembly" field exists (SPEC §2.2 note).
IMPORT_SIGNALS: tuple[str, ...] = ("imported", "unregistered", "auction sheet", "grade ", "jdm")

# Confidence tiers: a clean Assembly field is authoritative; text signals are weaker;
# the bare prior (most PK listings are locally assembled) is weakest.
CONF_CLEAN_FIELD = 0.95
CONF_TEXT_SIGNAL = 0.5
CONF_WEAK_PRIOR = 0.3


def derive_assembled_vs_imported(
    assembly_value: str | None, title: str, description: str
) -> tuple[str, float]:
    """Heuristic assembled-vs-imported + confidence flag (SPEC §2.2 note, §4).

    A clean "Assembly" field -> high confidence. Otherwise infer from title/description
    signals with lower confidence. Returns ``(label, confidence in [0, 1])``.
    """
    if assembly_value:
        val = assembly_value.strip().lower()
        if "import" in val:
            return "imported", CONF_CLEAN_FIELD
        if "local" in val:
            return "assembled", CONF_CLEAN_FIELD
    blob = f"{title} {description}".lower()
    if any(sig in blob for sig in IMPORT_SIGNALS):
        return "imported", CONF_TEXT_SIGNAL
    return "assembled", CONF_WEAK_PRIOR  # weak prior: most PK listings are locally assembled
