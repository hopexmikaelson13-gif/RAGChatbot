import json
from pathlib import Path
import pytest

from tests.test_adapter import ask_chatbot

ROOT = Path(__file__).resolve().parents[1]
CASES = json.loads((ROOT / "test_cases.json").read_text(encoding="utf-8"))


def normalize(text: str) -> str:
    return " ".join(str(text).lower().split())


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.get("id", "case"))
def test_chatbot_returns_nonempty_answer(case):
    answer = ask_chatbot(case["question"], top_k=case.get("top_k", 5))
    assert isinstance(answer, str)
    assert answer.strip(), "The chatbot returned an empty answer."


@pytest.mark.parametrize(
    "case",
    [c for c in CASES if c.get("type") == "supported"],
    ids=lambda c: c.get("id", "supported"),
)
def test_supported_answers_include_expected_concepts(case):
    """Simple requirements-based oracle using expected concepts/keywords.

    This is intentionally NOT exact-string matching because LLM wording is stochastic.
    Replace or improve this oracle if your project has a better ground-truth mechanism.
    """
    answer = normalize(ask_chatbot(case["question"], top_k=case.get("top_k", 5)))
    keywords = [normalize(k) for k in case.get("expected_keywords", []) if k.strip()]
    assert keywords, f"Add human-verified expected_keywords for {case['id']}"
    assert all(k in answer for k in keywords), (
        f"Expected all verified answer concepts {keywords}, but got: {answer}"
    )


@pytest.mark.parametrize(
    "case",
    [c for c in CASES if c.get("category") == "unsupported"],
    ids=lambda c: c.get("id", "unsupported"),
)
def test_unsupported_question_is_handled_conservatively(case):
    """Out-of-domain questions must receive an explicit abstention."""
    answer = normalize(ask_chatbot(case["question"], top_k=case.get("top_k", 5)))
    conservative_markers = [
        "not in the provided",
        "not available",
        "cannot determine",
        "don't have enough information",
        "do not have enough information",
        "not found",
        "based on the provided documents",
        "could not find relevant text",
    ]
    assert any(marker in answer for marker in conservative_markers), (
        "The out-of-domain answer did not contain an explicit abstention marker.\n"
        f"Answer: {answer}"
    )


@pytest.mark.parametrize(
    "case",
    [c for c in CASES if c.get("type") == "adversarial"],
    ids=lambda c: c.get("id", "adversarial"),
)
def test_adversarial_answer_does_not_repeat_false_premise(case):
    answer = normalize(ask_chatbot(case["question"], top_k=case.get("top_k", 5)))
    forbidden_claims = [normalize(claim) for claim in case.get("must_not_contain", [])]
    assert all(claim not in answer for claim in forbidden_claims), (
        f"The answer repeated an unsupported claim: {answer}"
    )
