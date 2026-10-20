"""Stage 2 claim predictor — every model_type trains, calibrates and scores."""
import numpy as np
import pytest

from app.ml import train_claim_predictor as tcp
from app.ml.seed_stage2_training_data import generate_training_data
from app.ml.stage2_features import STAGE2_FEATURE_COLS


@pytest.fixture(scope="module")
def df():
    return generate_training_data(n=2000)


@pytest.fixture
def artifact_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(tcp, "_ML_MODELS_DIR", tmp_path)
    monkeypatch.setattr(tcp, "_ARTIFACT_PATH", tmp_path / "claim_predictor.pkl")
    monkeypatch.setattr(tcp, "_META_PATH", tmp_path / "claim_predictor_meta.json")
    return tmp_path


@pytest.mark.parametrize("model_type", tcp.MODEL_TYPES)
def test_model_type_trains_and_scores(df, artifact_dir, model_type):
    result = tcp.train_model(df, model_type=model_type, n_estimators=50)

    assert result["model_type"] == model_type
    assert result["use_smote"] is False
    for key in ("auc_roc", "pr_auc", "precision_at_top_10pct", "brier_calibrated"):
        assert 0.0 <= result[key] <= 1.0, key
    assert set(result["feature_importance"]) == set(STAGE2_FEATURE_COLS)
    assert all(v >= 0 for v in result["feature_importance"].values())

    model, scaler = tcp.load_model()
    X = tcp._apply_scaler(df[STAGE2_FEATURE_COLS].values.astype(float), scaler)
    proba = model.predict_proba(X)[:, 1]
    assert np.all((proba >= 0.0) & (proba <= 1.0))

    p = tcp.score_claim(df.iloc[0][STAGE2_FEATURE_COLS].to_dict())
    assert 0.0 <= p <= 1.0
    assert tcp.load_metadata()["model_type"] == model_type


def test_min_samples_leaf_default_depends_on_smote(df):
    no_smote = tcp.train_model(df, n_estimators=20, persist_artifact=False)
    smote = tcp.train_model(df, n_estimators=20, use_smote=True, persist_artifact=False)
    assert no_smote["params"]["min_samples_leaf"] == 20
    assert smote["params"]["min_samples_leaf"] == 1
    assert smote["use_smote"] is True


def test_unknown_model_type_rejected(df):
    with pytest.raises(ValueError):
        tcp.train_model(df, model_type="svm", persist_artifact=False)
