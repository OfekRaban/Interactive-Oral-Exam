from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class EvaluationResult:
    score: int
    feedback: str
    label: str


# ── Batch evaluation result types ────────────────────────────────────────────

@dataclass
class DirectScoringResult:
    student_id: str
    name: str
    score: int
    feedback: str
    label: str
    error: Optional[str] = None


@dataclass
class RubricCriterionResult:
    name: str
    score: float
    max_score: float
    feedback: str


@dataclass
class RubricEvaluationResult:
    student_id: str
    name: str
    criteria: List[RubricCriterionResult]
    total_score: float
    max_score: float
    normalized_score: float  # 0-100
    feedback: str
    label: str
    error: Optional[str] = None


@dataclass
class AnchorEvaluationResult:
    student_id: str
    name: str
    score: float
    closest_anchor: str
    feedback: str
    label: str
    anchor_comparisons: Dict[str, str] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class PairwiseResult:
    student_a_id: str
    student_b_id: str
    winner: str  # "A" | "B" | "tie"
    confidence: float  # 0.0–1.0
    reasoning: str
    batch_id: Optional[str] = None  # set when using batched pairwise mode


@dataclass
class GroupRankingResult:
    """One LLM group-ranking call: N answers ranked from best to worst."""
    batch_id: str
    group_id: str
    student_ids: List[str]           # all students sent to the LLM
    ranked_student_ids: List[str]    # best → worst as returned/validated
    reasoning: str
    raw_response: str
    error: Optional[str] = None


@dataclass
class TournamentRanking:
    student_id: str
    name: str
    rank: int
    wins: int
    losses: int
    ties: int
    points: float   # wins + 0.5 * ties
    total_comparisons: int
