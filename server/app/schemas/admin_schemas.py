"""Pydantic schemas for ML-model + training-config admin endpoints.

Moved out of routes/admin.py so multiple routes / services share the
same public contract.
"""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


# ── ML model versions ─────────────────────────────────────────────────────

class MLModelSummary(BaseModel):
    """Shape consumed by the Admin → ML Model screen (active model)."""
    version: str
    trained_at: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    f2_score: Optional[float] = None
    auc_roc: float
    decision_threshold: Optional[float] = None
    brier_raw: Optional[float] = None
    brier_calibrated: Optional[float] = None
    training_samples: int
    feature_importance: dict = {}


class MLModelVersionRead(BaseModel):
    """Full row read for the ML versions list/history."""
    version_id: str
    model_name: str
    version: str
    trained_at: str
    training_rows: int
    training_params: dict = {}
    accuracy: float
    precision_score: Optional[float] = None
    recall_score: Optional[float] = None
    f1_score: Optional[float] = None
    f2_score: Optional[float] = None
    auc_roc: Optional[float] = None
    decision_threshold: Optional[float] = None
    positive_rate: float
    feature_importance: dict = {}
    is_active: bool
    notes: str = ""


# ── ML training config (admin-editable hyperparameters) ──────────────────

_MAX_FEATURES_RE = r"^(sqrt|log2|none|0?\.\d+|1\.0)$"
_CLASS_WEIGHT_RE = r"^(none|balanced|balanced_subsample)$"
_CRITERION_RE = r"^(gini|entropy|log_loss)$"


class MLTrainingConfigRead(BaseModel):
    n_estimators: int
    max_depth: Optional[int] = None
    min_samples_split: int = 2
    min_samples_leaf: int
    max_features: Optional[str] = "sqrt"
    max_leaf_nodes: Optional[int] = None
    bootstrap: bool = True
    class_weight: Optional[str] = None
    criterion: str = "gini"
    decision_threshold_mode: str
    manual_threshold: Optional[float] = None
    min_auc_to_promote: Optional[float] = None
    evaluation_population_size: int = 1000
    team_audit_capacity: int = 50
    updated_at: str


class MLTrainingConfigUpdate(BaseModel):
    n_estimators: int = Field(ge=10, le=2000)
    max_depth: Optional[int] = Field(default=None, ge=1, le=100)
    min_samples_split: int = Field(default=2, ge=2, le=200)
    min_samples_leaf: int = Field(ge=1, le=200)
    max_features: Optional[str] = Field(default="sqrt", pattern=_MAX_FEATURES_RE)
    max_leaf_nodes: Optional[int] = Field(default=None, ge=2, le=10000)
    bootstrap: bool = True
    class_weight: Optional[str] = Field(default=None, pattern=_CLASS_WEIGHT_RE)
    criterion: str = Field(default="gini", pattern=_CRITERION_RE)
    decision_threshold_mode: str = Field(pattern="^(auto_f2|manual|capacity_bounded_f2)$")
    manual_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    min_auc_to_promote: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    # Operational capacity knobs (see MLTrainingConfig model docstring).
    evaluation_population_size: int = Field(default=1000, ge=1, le=10_000_000)
    team_audit_capacity: int = Field(default=50, ge=1, le=10_000_000)


class MLTrialResult(BaseModel):
    """Metrics from an experimental (non-persisted) training run. No version is
    written and provider scores / the live artifact are left untouched."""
    method: str
    params_used: dict = {}
    accuracy: float
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    f2_score: Optional[float] = None
    auc_roc: Optional[float] = None
    decision_threshold: Optional[float] = None
    positive_rate: float
    training_rows: int
    feature_importance: dict = {}
    # Operational capacity projection AT the chosen threshold. flagged_count is
    # round(mean(proba >= threshold) * evaluation_population_size) — it moves with
    # the cutoff; the rest derive from it against team_audit_capacity. Present on
    # every mode, not just capacity_*.
    evaluation_population_size: int = 1000
    team_audit_capacity: int = 50
    flagged_count: int = 0
    capacity_utilization_pct: float = 0.0
    is_over_capacity: bool = False
    backlog_count: int = 0


class MLCommitRequest(MLTrainingConfigUpdate):
    """Save the chosen hyperparameters as the current config and persist a new
    active model version trained with them."""
    notes: str = ""


# ── Stage 2 Claim Predictor ──────────────────────────────────────────────────

class MLStage2ConfigUpdate(BaseModel):
    """Tunable hyperparameters for the Stage 2 claim predictor trial/retrain.
    Field names match train_claim_predictor.train_model kwargs so it can be
    passed via **body.model_dump()."""
    n_estimators: int = Field(default=300, ge=10, le=2000)
    max_depth: Optional[int] = Field(default=None, ge=1, le=100)
    min_samples_split: int = Field(default=2, ge=2, le=200)
    # None → 20 without SMOTE, 1 with it (resolved in train_model).
    min_samples_leaf: Optional[int] = Field(default=None, ge=1, le=200)
    max_features: Optional[str] = Field(default="sqrt", pattern=_MAX_FEATURES_RE)
    max_leaf_nodes: Optional[int] = Field(default=None, ge=2, le=10000)
    bootstrap: bool = True
    class_weight: Optional[str] = Field(default=None, pattern=_CLASS_WEIGHT_RE)
    criterion: str = Field(default="gini", pattern=_CRITERION_RE)
    decision_threshold_mode: str = Field(default="auto_f2", pattern="^(auto_f2|manual)$")
    manual_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    calibration_method: str = Field(default="sigmoid", pattern="^(sigmoid|isotonic|none)$")
    model_type: str = Field(default="random_forest", pattern="^(random_forest|logistic|hist_gb)$")
    use_smote: bool = False


class MLStage2TrialResult(BaseModel):
    """Metrics from a Stage 2 training run (trial: nothing persisted)."""
    model_type: str = "random_forest"
    use_smote: bool = False
    features: List[str] = []
    params: dict = {}
    positive_rate: float
    training_rows: int
    accuracy: float
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    f2_score: Optional[float] = None
    auc_roc: Optional[float] = None
    pr_auc: Optional[float] = None
    precision_at_top_10pct: Optional[float] = None
    threshold: Optional[float] = None
    calibration_method: str = "none"
    brier_raw: Optional[float] = None
    brier_calibrated: Optional[float] = None
    feature_importance: dict = {}


class MLStage2Summary(MLStage2TrialResult):
    """The current (last-committed) Stage 2 model — read from the metadata sidecar."""
    trained_at: Optional[str] = None


# ── EMV capacity preview ─────────────────────────────────────────────────────

class CapacityCaseRow(BaseModel):
    """One open case that passed the confidence gate, ranked by EV."""
    rank: int                       # 1-based rank by EV (desc) among gate survivors
    case_id: str
    case_number: str
    provider: str                   # rendering provider name or NPI
    amount_at_risk: float           # dollars (total_overpayment_amount)
    probability: float              # calibrated RandomForest P(overpayment) ∈ [0,1]
    ev: float                       # amount_at_risk × probability (expected $)
    within_capacity: bool           # True if rank ≤ capacity (worked this month);
                                    # False = passed the gate but left on the table


class EvCurvePoint(BaseModel):
    """One point on the cumulative-EV curve: fraction of total EV captured by the
    top `n` cases (by EV). Downsampled server-side."""
    n: int
    capture_ratio: float


class CapacityPreviewResponse(BaseModel):
    """Two-stage capacity funnel over the live open worklist.

    Stage 1 — GATE: keep only cases whose calibrated-RF P(overpayment) ≥
      `probability_cutoff`. Below-gate cases are excluded (the model isn't
      confident they're overpayments) — this is the Recall/Precision cutoff.
    Stage 2 — RANK & CUT: rank the survivors by EV = probability × amount_at_risk
      (dollars-weighted), keep the top `capacity`; survivors beyond it are backlog
      "left on the table". `rows` shows the top `capacity` + a backlog preview so
      the effect of the cut is visible.
    """
    probability_cutoff: float
    capacity: int
    total_open_cases: int
    gate_passed: int                       # cases with prob ≥ cutoff
    gate_excluded: int                     # cases below the cutoff (not audited)
    within_capacity_count: int             # min(capacity, gate_passed)
    backlog_count: int                     # passed the gate but beyond capacity
    total_ev: float = 0.0                  # Σ EV of ALL gate survivors
    captured_ev: float = 0.0               # Σ EV of the within-capacity cases
    backlog_ev: float = 0.0                # Σ EV left on the table (all backlog)
    capture_ratio: float = 0.0             # captured_ev / total_ev (0–1)
    ev_cut_line: Optional[float] = None    # EV of the last case kept (the line)
    ev_curve: List[EvCurvePoint] = []      # cumulative capture ratio vs # cases
    rows: List[CapacityCaseRow] = []       # top capacity + backlog-preview survivors
