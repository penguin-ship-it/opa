"""Mechanistic Stage 2 simulator — shape, determinism, and that labels come from events."""
import numpy as np
import pytest

from app.ml.simulate_stage2_claims import DE_MINIMIS, generate_training_data, simulate
from app.ml.stage2_features import STAGE2_FEATURE_COLS, TARGET_COL


@pytest.fixture(scope="module")
def df():
    return simulate(n_claims=3000, seed=7)


def test_columns_and_ranges(df):
    assert list(df.columns[:6]) == STAGE2_FEATURE_COLS + [TARGET_COL]
    assert len(df) == 3000
    assert df["P_risk"].between(0, 1).all()
    assert df["F4"].between(0, 1).all() and df["F5"].between(0, 1).all()
    assert df["F7"].between(0, 10).all() and set(df["F8"].unique()) <= {0.0, 1.0}
    assert set(df[TARGET_COL].unique()) <= {0, 1}


def test_deterministic_per_seed():
    a, b = generate_training_data(n=800, seed=3), generate_training_data(n=800, seed=3)
    assert a.equals(b)
    assert not a.equals(generate_training_data(n=800, seed=4))


def test_rare_positive_class(df):
    assert 0.03 < df[TARGET_COL].mean() < 0.20


def test_true_label_follows_overpaid_dollars(df):
    assert (df["true_op"] == (df["overpaid_amount"] >= DE_MINIMIS).astype(int)).all()
    assert (df["overpaid_amount"] <= df["paid_amount"] + 1e-6).all()


def test_audit_is_imperfect_but_informative(df):
    caught = df.loc[df.true_op == 1, TARGET_COL].mean()
    false_recovery = df.loc[df.true_op == 0, TARGET_COL].mean()
    assert 0.5 < caught < 1.0
    assert false_recovery < 0.03


def test_p_risk_separates_scheme_claims(df):
    schemer = df.schemes.str.contains("upcode|unbundle|unit_inflate|duplicate|phantom")
    assert df.loc[schemer, "P_risk"].mean() > df.loc[df.schemes == "", "P_risk"].mean()


def test_return_lines_joins_to_claims():
    d, lines = simulate(n_claims=500, seed=1, return_lines=True)
    assert set(lines.claim_id) == set(d.claim_id)
    assert np.allclose(lines.groupby("claim_id").allowed_amount.sum().sort_index().values,
                       d.set_index("claim_id").paid_amount.sort_index().values, atol=0.5)  # per-line rounding
