import json
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Request, Query
from ..middleware.auth import require_role
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from pydantic import BaseModel, Field
from datetime import datetime, timedelta

from ..database import get_db
from ..dao.user_dao import UserDAO
from ..dao.excluded_provider_dao import ExcludedProviderDAO
from ..models.workflow import OpaUser, OpaCase, PrioritizationConfig, DetectorRuleConfig, AuditLog
from ..models.reference import (
    ReferenceDataFreshness, CptCode, IcdCode, DrgCode, ModifierCode,
    CptModifierMap, CptDxCoverage,
)
from ..schemas.case_schemas import UserRead, CPTCodeRead, ICDCodeRead, DRGCodeRead, ModifierCodeRead
from ..schemas.admin_schemas import (
    MLModelSummary,
    MLModelVersionRead,
    MLTrainingConfigRead,
    MLTrainingConfigUpdate,
    MLTrialResult,
    MLCommitRequest,
    CapacityPreviewResponse,
    MLStage2ConfigUpdate,
    MLStage2TrialResult,
    MLStage2Summary,
)
from ..services.ml_model_service import MLModelService, params_from_config
from ..services.capacity_service import compute_capacity_preview
from ..services.prioritization_service import (
    get_config as get_priority_config,
    recompute_open_cases,
)
from ..services import detector_rule_service

router = APIRouter(prefix="/api/admin", tags=["admin"], dependencies=[Depends(require_role("admin"))])


class UserCreate(BaseModel):
    username: str
    email: str
    full_name: str
    role: str


class ReferenceDataFreshnessRead(BaseModel):
    source_name: str
    last_updated: str
    next_due: str
    status: str
    affected_detectors: List[str] = []


def _user_to_read(u: OpaUser) -> UserRead:
    return UserRead(
        id=u.user_id,
        username=u.username,
        full_name=u.full_name,
        email=u.email,
        role=u.role,
        is_active=u.is_active,
    )


@router.get("/users", response_model=List[UserRead])
async def list_users(db: AsyncSession = Depends(get_db)) -> List[UserRead]:
    dao = UserDAO(db)
    users = await dao.get_all()
    return [_user_to_read(u) for u in users]


@router.patch("/users/{user_id}", response_model=UserRead)
async def update_user(
    user_id: str,
    body: dict,
    db: AsyncSession = Depends(get_db),
) -> UserRead:
    result = await db.execute(select(OpaUser).where(OpaUser.user_id == user_id))
    user = result.scalar_one_or_none()
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    if "is_active" in body:
        user.is_active = bool(body["is_active"])
    await db.flush()
    return _user_to_read(user)


@router.get("/reference-freshness", response_model=List[ReferenceDataFreshnessRead])
async def get_reference_freshness(db: AsyncSession = Depends(get_db)) -> List[ReferenceDataFreshnessRead]:
    result = await db.execute(select(ReferenceDataFreshness))
    records = result.scalars().all()
    out = []
    for r in records:
        try:
            detectors = json.loads(r.affected_detectors)
        except Exception:
            detectors = []
        out.append(ReferenceDataFreshnessRead(
            source_name=r.source_name,
            last_updated=r.last_refreshed_at,
            next_due=r.next_scheduled_refresh,
            status=r.status,
            affected_detectors=detectors,
        ))
    return out


@router.post("/reference-freshness/{source_name}/refresh", response_model=ReferenceDataFreshnessRead)
async def refresh_reference_source(
    source_name: str,
    db: AsyncSession = Depends(get_db),
) -> ReferenceDataFreshnessRead:
    result = await db.execute(
        select(ReferenceDataFreshness).where(ReferenceDataFreshness.source_name == source_name)
    )
    record = result.scalar_one_or_none()
    if record is None:
        raise HTTPException(status_code=404, detail=f"Source '{source_name}' not found")

    # Preserve the original refresh cadence
    try:
        last = datetime.fromisoformat(record.last_refreshed_at)
        nxt  = datetime.fromisoformat(record.next_scheduled_refresh)
        interval_days = max(7, (nxt - last).days)
    except Exception:
        interval_days = 30

    now = datetime.utcnow()
    record.last_refreshed_at      = now.isoformat()
    record.next_scheduled_refresh = (now + timedelta(days=interval_days)).isoformat()
    record.status                 = "fresh"
    await db.flush()

    try:
        detectors = json.loads(record.affected_detectors)
    except Exception:
        detectors = []

    return ReferenceDataFreshnessRead(
        source_name=record.source_name,
        last_updated=record.last_refreshed_at,
        next_due=record.next_scheduled_refresh,
        status=record.status,
        affected_detectors=detectors,
    )


@router.get("/model", response_model=MLModelSummary)
async def get_active_model(db: AsyncSession = Depends(get_db)) -> MLModelSummary:
    summary = await MLModelService(db).get_active_summary()
    if summary is None:
        raise HTTPException(status_code=404, detail="No active model found")
    return summary


@router.get("/ml-models", response_model=List[MLModelVersionRead])
async def list_ml_models(db: AsyncSession = Depends(get_db)) -> List[MLModelVersionRead]:
    return await MLModelService(db).list_versions()


@router.get("/training-config", response_model=MLTrainingConfigRead)
async def get_training_config(db: AsyncSession = Depends(get_db)) -> MLTrainingConfigRead:
    return await MLModelService(db).get_training_config()


@router.put("/training-config", response_model=MLTrainingConfigRead)
async def update_training_config(
    body: MLTrainingConfigUpdate,
    db: AsyncSession = Depends(get_db),
) -> MLTrainingConfigRead:
    try:
        return await MLModelService(db).update_training_config(body)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/model/retrain")
async def retrain_model(db: AsyncSession = Depends(get_db)) -> dict:
    """Run a synchronous retrain inside the request using the saved config.
    For larger datasets, prefer POST /api/ml/train (thread executor)."""
    from ..ml.seed_training_data import generate_training_data
    from ..ml.train_billing_variance import train_model

    svc = MLModelService(db)
    cfg = await svc.get_training_config()
    params = params_from_config(cfg)
    try:
        df = generate_training_data()
        result = train_model(df, params=params)
        await svc.write_provider_scores(result["provider_scores"])
        version_id = await svc.write_training_result(
            result,
            params,
            model_name=result.get("model_name", "billing_variance_classifier"),
        )
        return {"status": "success", "version_id": version_id, "metrics": {
            "accuracy": result["accuracy"],
            "precision": result.get("precision"),
            "recall": result.get("recall"),
            "f1_score": result.get("f1_score"),
            "f2_score": result.get("f2_score"),
            "auc_roc": result.get("auc_roc"),
            "decision_threshold": result.get("threshold"),
        }}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


@router.post("/model/trial", response_model=MLTrialResult)
async def trial_train(body: MLTrainingConfigUpdate) -> MLTrialResult:
    """Experimental training run with the supplied hyperparameters. Returns
    metrics + feature importances but persists NOTHING — no version row, no
    provider-score write, and the live model artifact is left untouched. The
    engineer can re-run this freely while tuning, then call /model/commit when
    satisfied. Same params produce the same model (fixed random_state)."""
    if body.decision_threshold_mode == "manual" and body.manual_threshold is None:
        raise HTTPException(status_code=400,
                            detail="manual_threshold is required when decision_threshold_mode='manual'")

    from ..ml.seed_training_data import generate_training_data
    from ..ml.train_billing_variance import train_model

    params = params_from_config(body)
    try:
        df = generate_training_data()
        result = train_model(df, params=params, persist_artifact=False)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Trial training failed: {str(e)}")

    return MLTrialResult(
        method=result.get("method", ""),
        params_used=result.get("params_used", params),
        accuracy=result["accuracy"],
        precision=result.get("precision"),
        recall=result.get("recall"),
        f1_score=result.get("f1_score"),
        f2_score=result.get("f2_score"),
        auc_roc=result.get("auc_roc"),
        decision_threshold=result.get("threshold"),
        positive_rate=result["positive_rate"],
        training_rows=result["training_rows"],
        feature_importance=result.get("feature_importance", {}),
        evaluation_population_size=result.get("evaluation_population_size", 1000),
        team_audit_capacity=result.get("team_audit_capacity", 150),
        flagged_count=result.get("flagged_count", 0),
        capacity_utilization_pct=result.get("capacity_utilization_pct", 0.0),
        is_over_capacity=result.get("is_over_capacity", False),
        backlog_count=result.get("backlog_count", 0),
    )


@router.get("/model/capacity-preview", response_model=CapacityPreviewResponse)
async def capacity_preview(
    capacity: Optional[int] = Query(
        None, ge=0, le=10_000_000,
        description="Monthly review budget (top-N by EV). Defaults to saved team_audit_capacity.",
    ),
    target_capture: Optional[float] = Query(
        None, ge=0.0, le=1.0,
        description="Target fraction of total EV to capture; derives the capacity needed. Overrides capacity.",
    ),
    probability_cutoff: float = Query(
        0.04, ge=0.0, le=1.0,
        description="Secondary confidence gate: keep cases with calibrated P ≥ this.",
    ),
    backlog_preview: int = Query(
        25, ge=0, le=500,
        description="How many below-the-line survivors to include so the cut is visible.",
    ),
    db: AsyncSession = Depends(get_db),
) -> CapacityPreviewResponse:
    """EV-first audit plan over the live open worklist.

    Gates by calibrated P(overpayment) ≥ `probability_cutoff`, ranks survivors by
    EV (P × amount_at_risk), and cuts by EITHER `capacity` (top-N → reports the EV
    capture ratio) or `target_capture` (a target EV fraction → reports the capacity
    needed). Returns the cut set + a backlog preview plus total/captured EV and the
    capture ratio. `capacity` defaults to the saved team_audit_capacity.
    """
    if capacity is None and target_capture is None:
        cfg = await MLModelService(db).get_training_config()
        capacity = cfg.team_audit_capacity
    return await compute_capacity_preview(
        db, capacity=capacity, target_capture=target_capture,
        probability_cutoff=probability_cutoff,
        backlog_preview=backlog_preview,
    )


@router.get("/model2", response_model=MLStage2Summary)
async def get_stage2_model() -> MLStage2Summary:
    """Current (last-committed) Stage 2 claim predictor — metrics + params, read
    from the metadata sidecar. 404 if it has never been committed."""
    from ..ml.train_claim_predictor import load_metadata
    meta = load_metadata()
    if not meta:
        raise HTTPException(status_code=404, detail="Stage 2 model not trained yet")
    return MLStage2Summary(**{k: v for k, v in meta.items() if k in MLStage2Summary.model_fields})


@router.post("/model2/trial", response_model=MLStage2TrialResult)
async def stage2_trial(body: MLStage2ConfigUpdate) -> MLStage2TrialResult:
    """Experimental Stage 2 training run. Returns metrics + feature importances
    but persists NOTHING — the live claim_predictor artifact and the per-claim
    scores are untouched."""
    from ..ml.simulate_stage2_claims import generate_training_data
    from ..ml.train_claim_predictor import train_model
    if body.decision_threshold_mode == "manual" and body.manual_threshold is None:
        raise HTTPException(status_code=400, detail="manual_threshold is required when decision_threshold_mode='manual'")
    try:
        result = train_model(generate_training_data(), persist_artifact=False, **body.model_dump())
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Stage 2 trial failed: {e}")
    return MLStage2TrialResult(**{k: v for k, v in result.items() if k in MLStage2TrialResult.model_fields})


@router.post("/model2/retrain")
async def stage2_retrain(body: MLStage2ConfigUpdate) -> dict:
    """Retrain Stage 2 for real: persist the artifact + metadata, then re-score
    every claim's overpayment_probability so the audit plan reflects the new
    model. Returns metrics + the number of claims re-scored."""
    from ..ml.simulate_stage2_claims import generate_training_data
    from ..ml.train_claim_predictor import train_model
    from ..ml.score_claims import run as score_claims
    if body.decision_threshold_mode == "manual" and body.manual_threshold is None:
        raise HTTPException(status_code=400, detail="manual_threshold is required when decision_threshold_mode='manual'")
    try:
        result = train_model(generate_training_data(), persist_artifact=True, **body.model_dump())
        claims_scored = score_claims()
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Stage 2 retrain failed: {e}")
    return {
        "status": "success",
        "claims_scored": claims_scored,
        "model_type": result.get("model_type"),
        "metrics": {k: result.get(k) for k in
                    ("auc_roc", "pr_auc", "precision_at_top_10pct", "f2_score",
                     "precision", "recall", "threshold", "brier_calibrated",
                     "positive_rate")},
    }


@router.post("/model/commit")
async def commit_model(body: MLCommitRequest, db: AsyncSession = Depends(get_db)) -> dict:
    """Save the chosen hyperparameters as the current config, retrain for real
    (persisting the artifact + provider scores), and insert a new active model
    version. Honors the min_auc_to_promote gate. Call this once a trial run
    looks good."""
    if body.decision_threshold_mode == "manual" and body.manual_threshold is None:
        raise HTTPException(status_code=400,
                            detail="manual_threshold is required when decision_threshold_mode='manual'")

    from ..ml.seed_training_data import generate_training_data
    from ..ml.train_billing_variance import train_model

    svc = MLModelService(db)
    # Persist the approved config (everything except the commit-only `notes`).
    cfg_update = MLTrainingConfigUpdate(**body.model_dump(exclude={"notes"}))
    try:
        await svc.update_training_config(cfg_update)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    params = params_from_config(cfg_update)
    try:
        df = generate_training_data()
        result = train_model(df, params=params, persist_artifact=True)
        providers_updated = await svc.write_provider_scores(result["provider_scores"])
        version_id = await svc.write_training_result(
            result,
            params,
            model_name=result.get("model_name", "billing_variance_classifier"),
            notes=body.notes,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

    return {
        "status": "success",
        "version_id": version_id,
        "providers_updated": providers_updated,
        "metrics": {
            "accuracy": result["accuracy"],
            "precision": result.get("precision"),
            "recall": result.get("recall"),
            "f1_score": result.get("f1_score"),
            "f2_score": result.get("f2_score"),
            "auc_roc": result.get("auc_roc"),
            "decision_threshold": result.get("threshold"),
        },
    }


@router.get("/cpt-codes", response_model=List[CPTCodeRead])
async def list_cpt_codes(db: AsyncSession = Depends(get_db)) -> List[CPTCodeRead]:
    result = await db.execute(select(CptCode).order_by(CptCode.code))
    return [
        CPTCodeRead(
            code=c.code,
            description=c.description,
            code_type=c.code_type,
            risk_level=c.value_tier[0].upper() if c.value_tier else "M",
            cms_rac_flag=c.requires_auth,
            specialty_typical=c.specialty_typical,
            typical_setting=c.typical_setting,
            applicable_settings=c.applicable_settings,
            is_add_on=c.is_add_on,
            global_period_days=c.global_period_days,
            risk_score=c.risk_score,
            audit_notes=c.audit_notes,
            source_authority=c.source_authority,
            source_document=c.source_document,
            last_reviewed_at=c.last_reviewed_at,
            data_confidence=c.data_confidence,
            rule_certainty=c.rule_certainty,
        )
        for c in result.scalars().all()
    ]


# ── Excluded providers (OIG LEIE) — read-only reference browse ────────────────

class ExcludedProviderRead(BaseModel):
    excluded_provider_id: str
    npi: str
    last_name: Optional[str] = None
    first_name: Optional[str] = None
    middle_name: Optional[str] = None
    business_name: Optional[str] = None
    general_category: Optional[str] = None
    specialty: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    exclusion_type: Optional[str] = None
    exclusion_date: Optional[str] = None
    reinstate_date: Optional[str] = None
    waiver_date: Optional[str] = None
    source: str


class ExcludedProviderListResponse(BaseModel):
    items: List[ExcludedProviderRead]
    total: int
    page: int
    page_size: int


@router.get("/excluded-providers", response_model=ExcludedProviderListResponse)
async def list_excluded_providers(
    search: Optional[str] = Query(None, description="Match on NPI or last/first/business name"),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
) -> ExcludedProviderListResponse:
    """Browse the OIG LEIE exclusion table (the reference data DET-08 screens
    against by NPI). Read-only — the table is rebuilt from the LEIE file at seed."""
    rows, total = await ExcludedProviderDAO(db).search(
        search, skip=(page - 1) * page_size, limit=page_size
    )
    return ExcludedProviderListResponse(
        items=[ExcludedProviderRead.model_validate(r, from_attributes=True) for r in rows],
        total=total,
        page=page,
        page_size=page_size,
    )


# ── Shared update schema for enrichment fields (all four code tables) ─────────

class CodeEnrichmentUpdate(BaseModel):
    audit_notes: Optional[str] = None
    clinical_criteria: Optional[str] = None         # DRG only — ignored by others
    typical_setting: Optional[str] = None           # ICD — inpatient|outpatient|both|ed|snf|irf|…
    applicable_settings: Optional[str] = None       # ICD — JSON array of all applicable settings
    typical_drg: Optional[str] = None              # ICD — soft ref to drg_codes.code
    valid_as_primary_dx: Optional[bool] = None      # ICD — False for inherently secondary codes
    data_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    data_confidence_notes: Optional[str] = None
    rule_certainty: Optional[str] = None             # mandatory | guideline | heuristic
    source_authority: Optional[str] = None
    source_document: Optional[str] = None
    source_url: Optional[str] = None
    last_reviewed_at: Optional[str] = None           # YYYY-MM-DD


@router.patch("/cpt-codes/{code}", response_model=CPTCodeRead)
async def update_cpt_code(
    code: str, body: CodeEnrichmentUpdate, db: AsyncSession = Depends(get_db)
) -> CPTCodeRead:
    res = await db.execute(select(CptCode).where(CptCode.code == code))
    row = res.scalar_one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail=f"CPT code '{code}' not found")
    for field, val in body.model_dump(exclude_none=True).items():
        if hasattr(row, field):
            setattr(row, field, val)
    await db.commit()
    await db.refresh(row)
    return CPTCodeRead(
        code=row.code, description=row.description, code_type=row.code_type,
        risk_level=row.value_tier[0].upper() if row.value_tier else "M",
        cms_rac_flag=row.requires_auth, specialty_typical=row.specialty_typical,
        is_add_on=row.is_add_on, global_period_days=row.global_period_days,
        typical_setting=row.typical_setting,
        applicable_settings=row.applicable_settings,
        risk_score=row.risk_score,
        audit_notes=row.audit_notes, source_authority=row.source_authority,
        source_document=row.source_document, last_reviewed_at=row.last_reviewed_at,
        data_confidence=row.data_confidence, rule_certainty=row.rule_certainty,
    )


@router.patch("/icd-codes/{code}", response_model=ICDCodeRead)
async def update_icd_code(
    code: str, body: CodeEnrichmentUpdate, db: AsyncSession = Depends(get_db)
) -> ICDCodeRead:
    res = await db.execute(select(IcdCode).where(IcdCode.code == code))
    row = res.scalar_one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail=f"ICD code '{code}' not found")
    for field, val in body.model_dump(exclude_none=True).items():
        if hasattr(row, field):
            setattr(row, field, val)
    await db.commit()
    await db.refresh(row)
    return ICDCodeRead(
        code=row.code, description=row.description, code_type=row.code_type,
        category=row.value_tier or "general", chapter=row.chapter,
        is_manifestation=row.is_manifestation, is_etiology=row.is_etiology,
        audit_notes=row.audit_notes, source_authority=row.source_authority,
        source_document=row.source_document, last_reviewed_at=row.last_reviewed_at,
        data_confidence=row.data_confidence, rule_certainty=row.rule_certainty,
    )


@router.patch("/drg-codes/{code}", response_model=DRGCodeRead)
async def update_drg_code(
    code: str, body: CodeEnrichmentUpdate, db: AsyncSession = Depends(get_db)
) -> DRGCodeRead:
    res = await db.execute(select(DrgCode).where(DrgCode.code == code))
    row = res.scalar_one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail=f"DRG code '{code}' not found")
    for field, val in body.model_dump(exclude_none=True).items():
        if hasattr(row, field):
            setattr(row, field, val)
    await db.commit()
    await db.refresh(row)
    return DRGCodeRead(
        code=row.code, description=row.description, drg_type=row.drg_type,
        mdc=row.mdc, mdc_description=row.mdc_description, weight=row.weight,
        geometric_mean_los=row.geometric_mean_los, arithmetic_mean_los=row.arithmetic_mean_los,
        is_surgical=row.is_surgical, effective_fy=row.effective_fy,
        mcc_drg=row.mcc_drg, base_drg=row.base_drg,
        typical_principal_dx=row.typical_principal_dx, typical_procedures=row.typical_procedures,
        clinical_criteria=row.clinical_criteria, audit_notes=row.audit_notes,
        source_authority=row.source_authority,
        source_document=row.source_document, last_reviewed_at=row.last_reviewed_at,
        data_confidence=row.data_confidence, rule_certainty=row.rule_certainty,
    )


@router.patch("/modifier-codes/{code}", response_model=ModifierCodeRead)
async def update_modifier_code(
    code: str, body: CodeEnrichmentUpdate, db: AsyncSession = Depends(get_db)
) -> ModifierCodeRead:
    res = await db.execute(select(ModifierCode).where(ModifierCode.code == code))
    row = res.scalar_one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail=f"Modifier code '{code}' not found")
    for field, val in body.model_dump(exclude_none=True).items():
        if hasattr(row, field):
            setattr(row, field, val)
    await db.commit()
    await db.refresh(row)
    return ModifierCodeRead(
        code=row.code, description=row.description, modifier_type=row.modifier_type,
        applies_to=row.applies_to, payment_impact=row.payment_impact,
        payment_factor=row.payment_factor, ncci_override=row.ncci_override,
        requires_documentation=row.requires_documentation,
        audit_risk_score=row.audit_risk_score, audit_notes=row.audit_notes,
        source_authority=row.source_authority, source_document=row.source_document,
        last_reviewed_at=row.last_reviewed_at, data_confidence=row.data_confidence,
        rule_certainty=row.rule_certainty,
    )


# ── CPT-DX coverage (join table) ──────────────────────────────────────────────

class CptDxCoverageRead(BaseModel):
    cpt_code: str
    icd_code: str
    coverage_type: str
    rationale: Optional[str]
    source_authority: Optional[str]
    source_document: Optional[str]
    last_reviewed_at: Optional[str]
    data_confidence: float
    rule_certainty: str


class CptDxCoverageCreate(BaseModel):
    icd_code: str
    coverage_type: str
    rationale: Optional[str] = None
    source_authority: Optional[str] = None
    source_document: Optional[str] = None
    source_url: Optional[str] = None
    last_reviewed_at: Optional[str] = None
    data_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    data_confidence_notes: Optional[str] = None
    rule_certainty: str = "guideline"


@router.get("/cpt-dx-coverage", response_model=List[CptDxCoverageRead])
async def list_cpt_dx_coverage(
    cpt_code: str, db: AsyncSession = Depends(get_db)
) -> List[CptDxCoverageRead]:
    res = await db.execute(
        select(CptDxCoverage)
        .where(CptDxCoverage.cpt_code == cpt_code)
        .order_by(CptDxCoverage.coverage_type, CptDxCoverage.icd_code)
    )
    return [
        CptDxCoverageRead(
            cpt_code=r.cpt_code, icd_code=r.icd_code, coverage_type=r.coverage_type,
            rationale=r.rationale, source_authority=r.source_authority,
            source_document=r.source_document, last_reviewed_at=r.last_reviewed_at,
            data_confidence=r.data_confidence, rule_certainty=r.rule_certainty,
        )
        for r in res.scalars().all()
    ]


@router.post("/cpt-dx-coverage/{cpt_code}", response_model=CptDxCoverageRead, status_code=201)
async def create_cpt_dx_coverage(
    cpt_code: str, body: CptDxCoverageCreate, db: AsyncSession = Depends(get_db)
) -> CptDxCoverageRead:
    existing = await db.get(CptDxCoverage, (cpt_code, body.icd_code))
    if existing:
        raise HTTPException(status_code=409, detail="Coverage rule already exists for this pair")
    row = CptDxCoverage(
        cpt_code=cpt_code, icd_code=body.icd_code, coverage_type=body.coverage_type,
        rationale=body.rationale, source_authority=body.source_authority,
        source_document=body.source_document, source_url=body.source_url,
        last_reviewed_at=body.last_reviewed_at, data_confidence=body.data_confidence,
        data_confidence_notes=body.data_confidence_notes, rule_certainty=body.rule_certainty,
    )
    db.add(row)
    await db.commit()
    return CptDxCoverageRead(
        cpt_code=row.cpt_code, icd_code=row.icd_code, coverage_type=row.coverage_type,
        rationale=row.rationale, source_authority=row.source_authority,
        source_document=row.source_document, last_reviewed_at=row.last_reviewed_at,
        data_confidence=row.data_confidence, rule_certainty=row.rule_certainty,
    )


@router.delete("/cpt-dx-coverage/{cpt_code}/{icd_code}", status_code=204)
async def delete_cpt_dx_coverage(
    cpt_code: str, icd_code: str, db: AsyncSession = Depends(get_db)
) -> None:
    row = await db.get(CptDxCoverage, (cpt_code, icd_code))
    if row is None:
        raise HTTPException(status_code=404, detail="Coverage rule not found")
    await db.delete(row)
    await db.commit()


# ── CPT-modifier map (join table) ─────────────────────────────────────────────

class CptModifierMapRead(BaseModel):
    cpt_code: str
    modifier_code: str
    payment_factor: Optional[float]
    ncci_override: bool
    notes: Optional[str]
    source_authority: Optional[str]
    source_document: Optional[str]
    last_reviewed_at: Optional[str]
    data_confidence: float
    rule_certainty: str


class CptModifierMapCreate(BaseModel):
    modifier_code: str
    payment_factor: Optional[float] = None
    ncci_override: bool = False
    notes: Optional[str] = None
    source_authority: Optional[str] = None
    source_document: Optional[str] = None
    source_url: Optional[str] = None
    last_reviewed_at: Optional[str] = None
    data_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    data_confidence_notes: Optional[str] = None
    rule_certainty: str = "mandatory"


@router.get("/cpt-modifier-map", response_model=List[CptModifierMapRead])
async def list_cpt_modifier_map(
    cpt_code: str, db: AsyncSession = Depends(get_db)
) -> List[CptModifierMapRead]:
    res = await db.execute(
        select(CptModifierMap)
        .where(CptModifierMap.cpt_code == cpt_code)
        .order_by(CptModifierMap.modifier_code)
    )
    return [
        CptModifierMapRead(
            cpt_code=r.cpt_code, modifier_code=r.modifier_code,
            payment_factor=r.payment_factor, ncci_override=r.ncci_override,
            notes=r.notes, source_authority=r.source_authority,
            source_document=r.source_document, last_reviewed_at=r.last_reviewed_at,
            data_confidence=r.data_confidence, rule_certainty=r.rule_certainty,
        )
        for r in res.scalars().all()
    ]


@router.post("/cpt-modifier-map/{cpt_code}", response_model=CptModifierMapRead, status_code=201)
async def create_cpt_modifier_map(
    cpt_code: str, body: CptModifierMapCreate, db: AsyncSession = Depends(get_db)
) -> CptModifierMapRead:
    existing = await db.get(CptModifierMap, (cpt_code, body.modifier_code))
    if existing:
        raise HTTPException(status_code=409, detail="Entry already exists for this CPT-modifier pair")
    row = CptModifierMap(
        cpt_code=cpt_code, modifier_code=body.modifier_code,
        payment_factor=body.payment_factor, ncci_override=body.ncci_override,
        notes=body.notes, source_authority=body.source_authority,
        source_document=body.source_document, source_url=body.source_url,
        last_reviewed_at=body.last_reviewed_at, data_confidence=body.data_confidence,
        data_confidence_notes=body.data_confidence_notes, rule_certainty=body.rule_certainty,
    )
    db.add(row)
    await db.commit()
    return CptModifierMapRead(
        cpt_code=row.cpt_code, modifier_code=row.modifier_code,
        payment_factor=row.payment_factor, ncci_override=row.ncci_override,
        notes=row.notes, source_authority=row.source_authority,
        source_document=row.source_document, last_reviewed_at=row.last_reviewed_at,
        data_confidence=row.data_confidence, rule_certainty=row.rule_certainty,
    )


@router.delete("/cpt-modifier-map/{cpt_code}/{modifier_code}", status_code=204)
async def delete_cpt_modifier_map(
    cpt_code: str, modifier_code: str, db: AsyncSession = Depends(get_db)
) -> None:
    row = await db.get(CptModifierMap, (cpt_code, modifier_code))
    if row is None:
        raise HTTPException(status_code=404, detail="Modifier map entry not found")
    await db.delete(row)
    await db.commit()


@router.get("/drg-codes", response_model=List[DRGCodeRead])
async def list_drg_codes(db: AsyncSession = Depends(get_db)) -> List[DRGCodeRead]:
    result = await db.execute(select(DrgCode).order_by(DrgCode.code))
    return [
        DRGCodeRead(
            code=d.code,
            description=d.description,
            drg_type=d.drg_type,
            mdc=d.mdc,
            mdc_description=d.mdc_description,
            weight=d.weight,
            geometric_mean_los=d.geometric_mean_los,
            arithmetic_mean_los=d.arithmetic_mean_los,
            is_surgical=d.is_surgical,
            effective_fy=d.effective_fy,
            mcc_drg=d.mcc_drg,
            base_drg=d.base_drg,
            typical_principal_dx=d.typical_principal_dx,
            typical_procedures=d.typical_procedures,
            clinical_criteria=d.clinical_criteria,
            audit_notes=d.audit_notes,
            source_authority=d.source_authority,
            source_document=d.source_document,
            last_reviewed_at=d.last_reviewed_at,
            data_confidence=d.data_confidence,
            rule_certainty=d.rule_certainty,
        )
        for d in result.scalars().all()
    ]


@router.get("/modifier-codes", response_model=List[ModifierCodeRead])
async def list_modifier_codes(db: AsyncSession = Depends(get_db)) -> List[ModifierCodeRead]:
    result = await db.execute(select(ModifierCode).order_by(ModifierCode.code))
    return [
        ModifierCodeRead(
            code=m.code,
            description=m.description,
            modifier_type=m.modifier_type,
            applies_to=m.applies_to,
            payment_impact=m.payment_impact,
            payment_factor=m.payment_factor,
            ncci_override=m.ncci_override,
            requires_documentation=m.requires_documentation,
            audit_risk_score=m.audit_risk_score,
            audit_notes=m.audit_notes,
            source_authority=m.source_authority,
            source_document=m.source_document,
            last_reviewed_at=m.last_reviewed_at,
            data_confidence=m.data_confidence,
            rule_certainty=m.rule_certainty,
        )
        for m in result.scalars().all()
    ]


class PrioritizationConfigRead(BaseModel):
    severity_weight: float
    urgency_weight: float
    amount_cap: float
    rule_leak: float
    urgency_window_days: int
    high_threshold: float
    medium_threshold: float
    updated_at: str


class PrioritizationConfigUpdate(BaseModel):
    severity_weight: float = Field(ge=0.0, le=1.0)
    urgency_weight: float = Field(ge=0.0, le=1.0)
    amount_cap: float = Field(gt=0.0)
    rule_leak: float = Field(ge=0.0, le=0.5)
    urgency_window_days: int = Field(ge=1, le=365)
    high_threshold: float = Field(ge=0.0, le=100.0)
    medium_threshold: float = Field(ge=0.0, le=100.0)


def _cfg_to_read(c: PrioritizationConfig) -> PrioritizationConfigRead:
    return PrioritizationConfigRead(
        severity_weight=c.severity_weight,
        urgency_weight=c.urgency_weight,
        amount_cap=c.amount_cap,
        rule_leak=c.rule_leak,
        urgency_window_days=c.urgency_window_days,
        high_threshold=c.high_threshold,
        medium_threshold=c.medium_threshold,
        updated_at=c.updated_at,
    )


@router.get("/prioritization-config", response_model=PrioritizationConfigRead)
async def get_prioritization_config(db: AsyncSession = Depends(get_db)) -> PrioritizationConfigRead:
    cfg = await get_priority_config(db)
    return _cfg_to_read(cfg)


@router.put("/prioritization-config", response_model=PrioritizationConfigRead)
async def update_prioritization_config(
    body: PrioritizationConfigUpdate,
    db: AsyncSession = Depends(get_db),
) -> PrioritizationConfigRead:
    weight_sum = body.severity_weight + body.urgency_weight
    if abs(weight_sum - 1.0) > 0.001:
        raise HTTPException(
            status_code=400,
            detail=f"Severity + Urgency weights must sum to 1.0 (got {weight_sum:.3f})",
        )
    if body.high_threshold <= body.medium_threshold:
        raise HTTPException(
            status_code=400,
            detail="HIGH threshold must be greater than MEDIUM threshold",
        )

    cfg = await get_priority_config(db)
    cfg.severity_weight     = body.severity_weight
    cfg.urgency_weight      = body.urgency_weight
    cfg.amount_cap          = body.amount_cap
    cfg.rule_leak           = body.rule_leak
    cfg.urgency_window_days = body.urgency_window_days
    cfg.high_threshold      = body.high_threshold
    cfg.medium_threshold    = body.medium_threshold
    await db.flush()
    return _cfg_to_read(cfg)


@router.get("/prioritization-config/affected-count")
async def get_affected_case_count(db: AsyncSession = Depends(get_db)) -> dict:
    result = await db.execute(
        select(func.count()).select_from(OpaCase).where(OpaCase.is_active == True)  # noqa: E712
    )
    return {"open_cases": int(result.scalar_one())}


@router.post("/prioritization-config/recompute")
async def recompute_priorities(db: AsyncSession = Depends(get_db)) -> dict:
    cfg = await get_priority_config(db)
    summary = await recompute_open_cases(db, cfg)
    return {"status": "success", **summary}


class DetectorRuleRead(BaseModel):
    rule_code: str
    name: str
    description: str
    enabled_prepay: bool
    enabled_postpay: bool
    score: float
    updated_at: str
    has_implementation: bool
    layer: Optional[str]
    layer_order: Optional[int]
    applies_to: Optional[str]
    prepay: bool
    postpay: bool
    rationale: Optional[str]


class DetectorRuleUpdate(BaseModel):
    enabled_prepay: Optional[bool] = None
    enabled_postpay: Optional[bool] = None
    score: Optional[float] = Field(default=None, ge=0.0, le=1.0)


def _rule_to_read(r: DetectorRuleConfig) -> DetectorRuleRead:
    return DetectorRuleRead(
        rule_code=r.rule_code,
        name=r.name,
        description=r.description,
        enabled_prepay=r.enabled_prepay,
        enabled_postpay=r.enabled_postpay,
        score=r.score,
        updated_at=r.updated_at,
        has_implementation=r.has_implementation,
        layer=r.layer,
        layer_order=r.layer_order,
        applies_to=r.applies_to,
        prepay=r.prepay,
        postpay=r.postpay,
        rationale=r.rationale,
    )


async def _resolve_actor(request: Request, db: AsyncSession) -> str:
    """Return a real opa_users.user_id. Falls back to system.bot if header is missing/invalid."""
    header_id = request.headers.get("X-User-Id")
    if header_id:
        result = await db.execute(select(OpaUser.user_id).where(OpaUser.user_id == header_id))
        if result.scalar_one_or_none() is not None:
            return header_id
    result = await db.execute(select(OpaUser.user_id).where(OpaUser.username == "system.bot"))
    sys_id = result.scalar_one_or_none()
    if sys_id is not None:
        return sys_id
    # Last resort: any active user
    result = await db.execute(select(OpaUser.user_id).limit(1))
    return result.scalar_one()


@router.get("/detector-rules", response_model=List[DetectorRuleRead])
async def list_detector_rules(db: AsyncSession = Depends(get_db)) -> List[DetectorRuleRead]:
    rules = await detector_rule_service.get_all(db)
    return [_rule_to_read(r) for r in rules]


@router.put("/detector-rules/{rule_code}", response_model=DetectorRuleRead)
async def update_detector_rule(
    rule_code: str,
    body: DetectorRuleUpdate,
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> DetectorRuleRead:
    rule = await detector_rule_service.get_by_code(db, rule_code)
    if rule is None:
        raise HTTPException(status_code=404, detail=f"Rule '{rule_code}' not found")

    changes: dict = {}
    if body.enabled_prepay is not None and body.enabled_prepay != rule.enabled_prepay:
        changes["enabled_prepay"] = {"from": rule.enabled_prepay, "to": body.enabled_prepay}
        rule.enabled_prepay = body.enabled_prepay
    if body.enabled_postpay is not None and body.enabled_postpay != rule.enabled_postpay:
        changes["enabled_postpay"] = {"from": rule.enabled_postpay, "to": body.enabled_postpay}
        rule.enabled_postpay = body.enabled_postpay
    if body.score is not None and body.score != rule.score:
        changes["score"] = {"from": rule.score, "to": body.score}
        rule.score = body.score

    if not changes:
        return _rule_to_read(rule)

    actor_id = await _resolve_actor(request, db)
    rule.updated_by_user_id = actor_id

    db.add(AuditLog(
        case_id=None,
        actor_user_id=actor_id,
        action="rule_config_updated",
        from_state=None,
        to_state=None,
        reason=None,
        meta_json=json.dumps({"rule_code": rule_code, "changes": changes}),
    ))
    await db.flush()
    # A toggle or weight change alters the enabled set — drop the cached counts
    # so the next claim view reflects the new rule config.
    detector_rule_service.invalidate_rule_cache()
    return _rule_to_read(rule)


@router.get("/icd-codes", response_model=List[ICDCodeRead])
async def list_icd_codes(db: AsyncSession = Depends(get_db)) -> List[ICDCodeRead]:
    result = await db.execute(select(IcdCode).order_by(IcdCode.code))
    return [
        ICDCodeRead(
            code=c.code,
            description=c.description,
            code_type=c.code_type,
            category=c.value_tier or "general",
            chapter=c.chapter,
            is_manifestation=c.is_manifestation,
            is_etiology=c.is_etiology,
            typical_setting=c.typical_setting,
            applicable_settings=c.applicable_settings,
            typical_drg=c.typical_drg,
            valid_as_primary_dx=c.valid_as_primary_dx,
            audit_notes=c.audit_notes,
            source_authority=c.source_authority,
            source_document=c.source_document,
            last_reviewed_at=c.last_reviewed_at,
            data_confidence=c.data_confidence,
            rule_certainty=c.rule_certainty,
        )
        for c in result.scalars().all()
    ]
