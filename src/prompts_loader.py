"""Load frozen RAG prompt templates from ``data/prompts.json``.

This module is the single source of truth for the three generation prompts
(``prompt_zero_shot``, ``prompt_one_shot``, ``prompt_few_shot``). The JSON
artifact freezes the prompt text so evaluation and serving code no longer
require a live Langfuse instance. The path is resolved relative to this file
so loading works from any working directory.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate

_PROMPTS_PATH = Path(__file__).resolve().parents[1] / "data" / "prompts.json"

_PROMPT_KEYS = ("prompt_zero_shot", "prompt_one_shot", "prompt_few_shot")
_REQUIRED_PLACEHOLDERS = ("{context}", "{question}")


@lru_cache(maxsize=1)
def load_prompt_texts() -> dict[str, str]:
    """Return the raw template strings keyed by prompt artifact name.

    Reads the frozen prompts file once and caches the result.

    Raises:
        RuntimeError: If the file is missing or unreadable, or if any expected
            key is absent, not a string, or missing a required placeholder
            (``{context}`` / ``{question}``). The message names the offending
            file or key.
    """
    if not _PROMPTS_PATH.is_file():
        raise RuntimeError(f"Frozen prompts file not found: {_PROMPTS_PATH}")
    try:
        raw = json.loads(_PROMPTS_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError(
            f"Cannot read frozen prompts file {_PROMPTS_PATH}: {exc}"
        ) from exc
    if not isinstance(raw, dict):
        raise RuntimeError(
            f"Frozen prompts file {_PROMPTS_PATH} must contain a JSON object"
        )

    # The artifact nests the templates under a top-level "prompts" object;
    # fall back to a flat mapping if that key is absent.
    source = raw.get("prompts")
    if not isinstance(source, dict):
        source = raw

    texts: dict[str, str] = {}
    for key in _PROMPT_KEYS:
        value = source.get(key)
        if not isinstance(value, str) or not value:
            raise RuntimeError(
                f"Missing or invalid prompt key {key!r} in {_PROMPTS_PATH}"
            )
        for placeholder in _REQUIRED_PLACEHOLDERS:
            if placeholder not in value:
                raise RuntimeError(
                    f"Prompt {key!r} in {_PROMPTS_PATH} is missing "
                    f"required placeholder {placeholder}"
                )
        texts[key] = value
    return texts


@lru_cache(maxsize=1)
def load_prompts() -> dict[str, ChatPromptTemplate]:
    """Build a ``ChatPromptTemplate`` for each frozen prompt key.

    Raises:
        RuntimeError: Propagated from :func:`load_prompt_texts` when the JSON
            artifact is missing or a key is absent / malformed.
    """
    return {
        key: ChatPromptTemplate.from_template(text)
        for key, text in load_prompt_texts().items()
    }


@lru_cache(maxsize=None)
def get_local_prompt(name: str) -> ChatPromptTemplate:
    """Return the frozen template registered under ``name``.

    Args:
        name: Full prompt artifact name, e.g. ``prompt_zero_shot``.

    Returns:
        The cached ``ChatPromptTemplate`` for ``name``.

    Raises:
        KeyError: If ``name`` is not one of the frozen prompt keys; the
            message lists the known names.
    """
    prompts = load_prompts()
    if name not in prompts:
        raise KeyError(
            f"Unknown prompt name {name!r}; expected one of: "
            f"{', '.join(sorted(prompts))}"
        )
    return prompts[name]
