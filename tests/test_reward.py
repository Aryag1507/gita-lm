"""
Unit tests for reward model scoring logic.

Tests the heuristic_score function and the rerank_candidates selection logic.
rerank uses a fake scorer so no GPT-2 download is needed.
"""
import pytest

from src.reward.reward_model import heuristic_score, rerank_candidates


def test_empty_text_scores_zero():
    assert heuristic_score("") == 0.0
    assert heuristic_score("    ") == 0.0


def test_score_in_unit_range():
    text = "Krishna is the Supreme Lord and the devotee engages in bhakti yoga."
    score = heuristic_score(text)
    assert 0.0 <= score <= 1.0


def test_theological_text_scores_higher_than_generic():
    """Text dense in Prabhupada vocabulary should outscore generic prose."""
    theological = (
        "Krishna the Supreme Lord teaches Arjuna about the eternal soul, "
        "devotional service, bhakti yoga, and transcendental consciousness. "
        "The conditioned soul under material energy attains liberation through "
        "Krishna consciousness and surrender to the Supreme."
    )
    generic = (
        "The weather today is quite nice and I went to the store to buy some "
        "groceries before heading home to cook dinner for my family tonight."
    )
    assert heuristic_score(theological) > heuristic_score(generic)


def test_citation_and_sanskrit_boost_structure_score():
    with_markers = "As stated in BG 2.47, the ātmā is eternal. Krishna soul bhakti."
    without_markers = "This is a plain sentence with Krishna soul bhakti words only."
    assert heuristic_score(with_markers) > heuristic_score(without_markers)


def test_rerank_selects_highest_scoring_candidate():
    candidates = ["bad candidate", "best candidate", "mid candidate"]
    score_map = {
        "bad candidate": 0.1,
        "best candidate": 0.9,
        "mid candidate": 0.5,
    }

    import torch

    class FakeModel:
        def eval(self): return self
        def __call__(self, input_ids, attention_mask):
            return torch.tensor(FakeModel.pending)

    class FakeTok:
        def __call__(self, text, **kwargs):
            FakeModel.pending = score_map[text]
            return {
                "input_ids": torch.zeros(1, 4, dtype=torch.long),
                "attention_mask": torch.ones(1, 4, dtype=torch.long),
            }

    best, scores = rerank_candidates(candidates, FakeModel(), FakeTok(), "cpu")

    assert best == "best candidate"
    assert scores == pytest.approx([0.1, 0.9, 0.5], abs=1e-6)
    assert max(scores) == scores[1]
