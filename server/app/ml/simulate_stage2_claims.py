"""simulate_stage2_claims.py — mechanistic claim simulator for the Stage 2 model.

Unlike seed_stage2_training_data (which samples the five features directly and
draws the label from a hand-written logit), this module simulates the *world*
and lets the features and labels fall out of it:

  1. A provider population: specialty, claim volume (heavy-tailed), and a latent
     behaviour — honest, sloppy (random coding errors), or running one or two
     overpayment schemes at a random intensity. Some providers also sit on a
     mis-loaded payer contract (a payer-side error the provider didn't cause).
  2. Claims: service lines drawn from a specialty CPT menu, with legitimate
     modifiers, specialty-typical units, and allowed = fee schedule × units.
  3. Schemes act on individual claims and record the dollars they add:
       upcode        — E/M / psychotherapy code bumped one level
       unbundle      — extra add-on line billed with modifier 59/25
       unit_inflate  — units ×2-4 on time/unit-based codes
       duplicate     — the same line billed twice
       phantom       — a fabricated high-value claim (fully overpaid)
     Payer-side errors: contract mis-load (allowed above fee on every line),
     modifier pricing errors (some modifier lines paid above fee), and
     "hidden" eligibility/COB errors that leave no trace in the claim lines.
  4. Labels: a claim is truly overpaid when its overpaid $ ≥ DE_MINIMIS. The
     observed label `confirmed_op` comes from an imperfect audit — big
     overpayments are easier to prove, small ones are missed, and a few clean
     claims get (wrongly) recovered.
  5. P_risk: a cross-fitted provider model trained on each provider's *prior*
     history (a separate simulated period), so it is a noisy learned estimate —
     never the provider's true behaviour, and never fit on the claims it scores.

F4/F5/F7/F8 are computed with stage2_features.compute_features on the
simulated lines, i.e. exactly the code path used for real claims at inference.

Public API:
    df = generate_training_data(n=6000, seed=42)      # STAGE2_FEATURE_COLS + target + extras
    df, lines = simulate(n_claims, seed, return_lines=True)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .stage2_features import (
    HIGH_RISK_CPTS, HIGH_RISK_MODIFIERS, STAGE2_FEATURE_COLS, TARGET_COL, compute_features,
)

DE_MINIMIS = 20.0  # overpaid $ below this isn't worth recovering → not an overpayment

# ── reference world ──────────────────────────────────────────────────────────
# code -> base rate ($). Illustrative Medicare-ish rates.
RATES = {
    "99212": 57, "99213": 92, "99214": 131, "99215": 184,
    "99283": 120, "99284": 200, "99285": 290,
    "90832": 70, "90834": 100, "90837": 150, "90792": 180,
    "93000": 17, "93010": 9, "93306": 220, "93015": 80, "93880": 200,
    "36415": 3, "81002": 4, "96372": 15, "90471": 25, "G0439": 130,
    "20610": 65, "73562": 40, "27447": 1400, "29881": 560, "97110": 30,
    "95910": 230, "95886": 90, "64483": 230, "62323": 220, "70553": 400,
    "11102": 100, "17000": 70, "17003": 5, "11042": 120,
    "92014": 130, "92012": 90, "92134": 40, "67028": 110, "J0178": 20,
    "96374": 60, "12001": 110,
}
UPCODE = {"99212": "99213", "99213": "99214", "99214": "99215",
          "99283": "99284", "99284": "99285", "90832": "90834", "90834": "90837",
          "92012": "92014"}
UNIT_CODES = {"97110": (1, 3), "17003": (1, 6), "J0178": (1, 2), "96372": (1, 2), "95886": (1, 2)}
ADDON_CODES = ["96372", "36415", "93000", "97110", "17003", "92134", "81002"]

# specialty -> (weight, E/M menu with weights, procedure menu with weights, mean extra lines)
SPECIALTIES: dict[str, dict[str, Any]] = {
    "Family Medicine":    dict(w=0.20, em={"99212": .15, "99213": .50, "99214": .30, "99215": .05},
                               proc={"36415": .3, "81002": .15, "93000": .15, "96372": .15, "90471": .1, "G0439": .15}, extra=0.7),
    "Internal Medicine":  dict(w=0.15, em={"99212": .10, "99213": .45, "99214": .37, "99215": .08},
                               proc={"36415": .35, "93000": .25, "96372": .15, "G0439": .25}, extra=0.7),
    "Cardiology":         dict(w=0.09, em={"99213": .35, "99214": .50, "99215": .15},
                               proc={"93000": .35, "93306": .3, "93015": .2, "93880": .15}, extra=0.9),
    "Orthopedic Surgery": dict(w=0.09, em={"99213": .45, "99214": .45, "99215": .10},
                               proc={"20610": .35, "73562": .3, "97110": .2, "29881": .1, "27447": .05}, extra=0.9),
    "Psychiatry":         dict(w=0.12, em={"90834": .45, "90837": .30, "90832": .15, "90792": .10},
                               proc={"99214": 1.0}, extra=0.15),
    "Neurology":          dict(w=0.07, em={"99213": .35, "99214": .50, "99215": .15},
                               proc={"95910": .3, "95886": .3, "64483": .15, "62323": .1, "70553": .15}, extra=0.8),
    "Dermatology":        dict(w=0.08, em={"99212": .25, "99213": .55, "99214": .20},
                               proc={"11102": .35, "17000": .35, "17003": .2, "11042": .1}, extra=1.0),
    "Ophthalmology":      dict(w=0.08, em={"92012": .45, "92014": .45, "99213": .10},
                               proc={"92134": .5, "67028": .3, "J0178": .2}, extra=1.0),
    "Emergency Medicine": dict(w=0.12, em={"99283": .35, "99284": .45, "99285": .20},
                               proc={"93010": .3, "96374": .4, "12001": .3}, extra=1.1),
}
SCHEMES = ["upcode", "unbundle", "unit_inflate", "duplicate", "phantom"]
SCHEME_WEIGHTS = np.array([0.34, 0.24, 0.16, 0.16, 0.10])


@dataclass
class Provider:
    npi: str
    specialty: str
    volume: float
    sloppy_rate: float = 0.0
    schemes: dict[str, float] = field(default_factory=dict)   # scheme -> P(applied to a claim)
    contract_markup: float = 0.0                              # payer mis-load, >0 = overpaid every line
    modifier_rate: float = 0.04                               # legit modifier usage


def _pick(rng: np.random.Generator, menu: dict[str, float]) -> str:
    keys = list(menu)
    p = np.array([menu[k] for k in keys], dtype=float)
    return keys[rng.choice(len(keys), p=p / p.sum())]


def _make_providers(n: int, rng: np.random.Generator) -> list[Provider]:
    specs = list(SPECIALTIES)
    sw = np.array([SPECIALTIES[s]["w"] for s in specs])
    out = []
    for i in range(n):
        p = Provider(npi=f"9{i:09d}", specialty=specs[rng.choice(len(specs), p=sw / sw.sum())],
                     volume=float(rng.lognormal(0.0, 0.9)))
        r = rng.random()
        if r < 0.15:                                   # sloppy coder
            p.sloppy_rate = float(rng.uniform(0.03, 0.10))
        elif r < 0.15 + float(rng.uniform(0.10, 0.16)):  # schemer (share itself random)
            k = 1 + int(rng.random() < 0.3)
            for s in rng.choice(SCHEMES, size=k, replace=False, p=SCHEME_WEIGHTS):
                p.schemes[str(s)] = float(rng.beta(2.0, 3.0))
            if "unbundle" in p.schemes:
                p.modifier_rate = float(rng.uniform(0.10, 0.30))
        if rng.random() < 0.04:
            p.contract_markup = float(rng.uniform(0.12, 0.40))
        p.modifier_rate += float(rng.uniform(0.0, 0.04))
        out.append(p)
    return out


def _line(code: str, units: float, fee_mult: float, mods: tuple[str, str] = ("", "")) -> dict[str, Any]:
    fee = RATES[code] * units * fee_mult
    return {"cpt_code": code, "modifier_1": mods[0], "modifier_2": mods[1], "units": float(units),
            "fee_schedule_amount": round(fee, 2), "allowed_amount": round(fee, 2), "overpaid": 0.0}


def _simulate_claim(p: Provider, rng: np.random.Generator) -> tuple[list[dict], dict[str, Any]]:
    spec = SPECIALTIES[p.specialty]
    fee_mult = 1.0  # one fee schedule per provider-org; ratios are what matter
    applied: list[str] = []

    if "phantom" in p.schemes and rng.random() < p.schemes["phantom"] * 0.35:
        # Fabricated claim: high-value codes, every dollar overpaid.
        code = _pick(rng, spec["proc"]) if rng.random() < 0.5 else max(spec["em"], key=lambda c: RATES[c])
        lines = [_line(code, 1, fee_mult)]
        if rng.random() < 0.5:
            lines.append(_line(max(spec["em"], key=lambda c: RATES[c]), 1, fee_mult))
        for ln in lines:
            ln["overpaid"] = ln["allowed_amount"]
        applied.append("phantom")
    else:
        em = _pick(rng, spec["em"])
        lines = [_line(em, 1, fee_mult)]
        for _ in range(rng.poisson(spec["extra"])):
            code = _pick(rng, spec["proc"])
            lo, hi = UNIT_CODES.get(code, (1, 1))
            units = int(rng.integers(lo, hi + 1))
            mods = (str(rng.choice(["59", "76", "GT", "95"])), "") if rng.random() < p.modifier_rate else ("", "")
            lines.append(_line(code, units, fee_mult, mods))
        if len(lines) > 1 and rng.random() < 0.6:   # significant, separate E/M on a procedure day
            lines[0]["modifier_1"] = "25"

        # Sloppy coders: honest-mistake upcodes.
        if p.sloppy_rate and rng.random() < p.sloppy_rate and lines[0]["cpt_code"] in UPCODE:
            _upcode(lines[0], fee_mult)
            applied.append("sloppy")

        for s, intensity in p.schemes.items():
            if s == "phantom" or rng.random() >= intensity:
                continue
            if s == "upcode":
                for ln in lines:
                    if ln["cpt_code"] in UPCODE:
                        _upcode(ln, fee_mult)
                        if rng.random() < 0.5:
                            break
                applied.append(s)
            elif s == "unbundle":
                code = str(rng.choice(ADDON_CODES))
                lo, hi = UNIT_CODES.get(code, (1, 1))
                ln = _line(code, int(rng.integers(lo, hi + 1)), fee_mult, (str(rng.choice(["59", "25"])), ""))
                ln["overpaid"] = ln["allowed_amount"]
                lines.append(ln)
                applied.append(s)
            elif s == "unit_inflate":
                cands = [ln for ln in lines if ln["cpt_code"] in UNIT_CODES]
                if not cands:  # add a unit-based line to inflate
                    code = str(rng.choice(list(UNIT_CODES)))
                    cands = [_line(code, 1, fee_mult)]
                    lines.append(cands[0])
                ln = cands[0]
                true_units = ln["units"]
                ln["units"] = float(true_units * int(rng.integers(2, 5)))
                ln["fee_schedule_amount"] = ln["allowed_amount"] = round(RATES[ln["cpt_code"]] * ln["units"] * fee_mult, 2)
                ln["overpaid"] += RATES[ln["cpt_code"]] * (ln["units"] - true_units) * fee_mult
                applied.append(s)
            elif s == "duplicate":
                dup = dict(lines[int(rng.integers(0, len(lines)))])
                dup["overpaid"] = dup["allowed_amount"]
                lines.append(dup)
                applied.append(s)

    # Payer-side pricing errors.
    for ln in lines:
        mods = {ln["modifier_1"], ln["modifier_2"]}
        markup = p.contract_markup
        if mods & HIGH_RISK_MODIFIERS and rng.random() < 0.06:
            markup += float(rng.uniform(0.10, 0.60))   # modifier priced above fee schedule
        if markup:
            extra = ln["fee_schedule_amount"] * markup
            ln["allowed_amount"] = round(ln["allowed_amount"] + extra, 2)
            ln["overpaid"] += extra
            if "payer_pricing" not in applied:
                applied.append("payer_pricing")

    paid = sum(ln["allowed_amount"] for ln in lines)
    overpaid = sum(ln["overpaid"] for ln in lines)
    if rng.random() < 0.025:   # eligibility / COB error — invisible in the lines
        overpaid = paid
        applied.append("hidden_eligibility")
    return lines, {"paid_amount": round(paid, 2), "overpaid_amount": round(min(overpaid, paid), 2),
                   "schemes": ",".join(applied)}


def _upcode(ln: dict, fee_mult: float) -> None:
    old = ln["cpt_code"]
    new = UPCODE[old]
    delta = (RATES[new] - RATES[old]) * ln["units"] * fee_mult
    ln["cpt_code"] = new
    ln["fee_schedule_amount"] = round(ln["fee_schedule_amount"] + delta, 2)
    ln["allowed_amount"] = round(ln["allowed_amount"] + delta, 2)
    ln["overpaid"] += delta


def _audit(overpaid: float, rng: np.random.Generator) -> int:
    """Imperfect audit → observed label. Bigger overpayments are easier to prove."""
    if overpaid >= DE_MINIMIS:
        return int(rng.random() < min(0.95, 0.55 + 0.40 * min(1.0, overpaid / 300.0)))
    return int(rng.random() < 0.01)   # erroneous recovery


def _simulate_period(providers: list[Provider], n_claims: int, rng: np.random.Generator):
    vol = np.array([p.volume for p in providers])
    counts = rng.multinomial(n_claims, vol / vol.sum())
    rows, all_lines = [], []
    cid = 0
    for p, k in zip(providers, counts):
        for _ in range(int(k)):
            lines, meta = _simulate_claim(p, rng)
            rows.append({"claim_id": cid, "provider_npi": p.npi, "specialty": p.specialty,
                         "lines": lines, **meta})
            cid += 1
    return rows


# ── Stage 1 stand-in: provider risk learned from prior history ───────────────

def _provider_aggregates(claims: list[dict]) -> dict[str, float]:
    lines = [ln for c in claims for ln in c["lines"]]
    n_l = max(len(lines), 1)
    return {
        "avg_units_per_line": float(np.mean([ln["units"] for ln in lines])) if lines else 0.0,
        "high_value_cpt_ratio": sum(ln["cpt_code"] in HIGH_RISK_CPTS for ln in lines) / n_l,
        "multi_line_claim_ratio": float(np.mean([len(c["lines"]) > 1 for c in claims])),
        "modifier_usage_rate": sum(bool({ln["modifier_1"], ln["modifier_2"]} & HIGH_RISK_MODIFIERS) for ln in lines) / n_l,
        "same_day_multi_cpt_rate": float(np.mean([len({ln["cpt_code"] for ln in c["lines"]}) > 1 for c in claims])),
        "allowed_per_claim": float(np.mean([c["paid_amount"] for c in claims])),
    }


def _fit_p_risk(providers: list[Provider], history: list[dict], rng: np.random.Generator) -> dict[str, float]:
    """Cross-fitted provider risk. Features from history half 1 (incl. the
    confirmed rate on its audited sample); label = confirmed rate on the audited
    sample of half 2 ≥ 8%. Out-of-fold predictions → P_risk."""
    by_p: dict[str, list[dict]] = {p.npi: [] for p in providers}
    for c in history:
        by_p[c["provider_npi"]].append(c)

    feats, labels, npis = [], [], []
    for p in providers:
        cl = by_p[p.npi]
        if len(cl) < 4:
            continue
        rng.shuffle(cl)
        h1, h2 = cl[: len(cl) // 2], cl[len(cl) // 2:]
        f = _provider_aggregates(h1)
        a1 = [c for c in h1 if rng.random() < 0.3] or h1[:1]       # audit sample
        f["prior_overpayment_rate"] = float(np.mean([_audit(c["overpaid_amount"], rng) for c in a1]))
        a2 = [c for c in h2 if rng.random() < 0.3] or h2[:2]
        labels.append(int(np.mean([_audit(c["overpaid_amount"], rng) for c in a2]) >= 0.08))
        feats.append(f)
        npis.append(p.npi)

    X = pd.DataFrame(feats)
    # specialty peer deviation of allowed/claim (z-score within specialty)
    spec = pd.Series({p.npi: p.specialty for p in providers})[npis].values
    g = X.groupby(spec)["allowed_per_claim"]
    X["specialty_peer_deviation"] = ((X["allowed_per_claim"] - g.transform("mean")) / g.transform("std").replace(0, 1)).fillna(0)
    X = X.drop(columns="allowed_per_claim")
    y = np.array(labels)

    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5))
    if len(np.unique(y)) < 2 or min(np.bincount(y)) < 5:
        base = float(y.mean()) if len(y) else 0.5
        risk = {n: base for n in npis}
    else:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=int(rng.integers(1 << 30)))
        proba = cross_val_predict(model, X.values, y, cv=cv, method="predict_proba")[:, 1]
        risk = dict(zip(npis, proba.tolist()))
    default = float(np.mean(list(risk.values()))) if risk else 0.5
    return {p.npi: float(risk.get(p.npi, default)) for p in providers}


# ── public API ───────────────────────────────────────────────────────────────

def simulate(n_claims: int = 6000, seed: int = 42, n_providers: int | None = None,
             history_ratio: float = 3.0, return_lines: bool = False):
    rng = np.random.default_rng(seed)
    n_providers = n_providers or max(40, n_claims // 20)
    providers = _make_providers(n_providers, rng)

    history = _simulate_period(providers, int(n_claims * history_ratio), rng)
    p_risk = _fit_p_risk(providers, history, rng)

    current = _simulate_period(providers, n_claims, rng)
    rows, line_rows = [], []
    for c in current:
        f = compute_features(p_risk[c["provider_npi"]], c["lines"], c["specialty"])
        true_op = int(c["overpaid_amount"] >= DE_MINIMIS)
        rows.append({**f, TARGET_COL: _audit(c["overpaid_amount"], rng),
                     "true_op": true_op, "claim_id": c["claim_id"], "provider_npi": c["provider_npi"],
                     "specialty": c["specialty"], "n_lines": len(c["lines"]),
                     "paid_amount": c["paid_amount"], "overpaid_amount": c["overpaid_amount"],
                     "schemes": c["schemes"]})
        if return_lines:
            line_rows += [{"claim_id": c["claim_id"], **{k: v for k, v in ln.items()}} for ln in c["lines"]]

    df = pd.DataFrame(rows)
    cols = STAGE2_FEATURE_COLS + [TARGET_COL] + [c for c in df.columns if c not in STAGE2_FEATURE_COLS + [TARGET_COL]]
    df = df[cols]
    return (df, pd.DataFrame(line_rows)) if return_lines else df


def generate_training_data(n: int = 6000, seed: int = 42) -> pd.DataFrame:
    """Drop-in replacement for seed_stage2_training_data.generate_training_data.
    Returns STAGE2_FEATURE_COLS + TARGET_COL first, then diagnostic extras
    (true_op, provider_npi, paid_amount, overpaid_amount, schemes, ...)."""
    return simulate(n_claims=n, seed=seed)


if __name__ == "__main__":
    d = simulate()
    print(f"rows={len(d)}  confirmed_op rate={d[TARGET_COL].mean():.3f}  true_op rate={d['true_op'].mean():.3f}")
    print(d[STAGE2_FEATURE_COLS].describe().round(3).to_string())
    print("\nP_risk by provider true behaviour proxy (any scheme on claim):")
    print(d.assign(scheme=d.schemes.str.split(",").str[0].replace("", "none")).groupby("scheme")
          .agg(n=("P_risk", "size"), p_risk=("P_risk", "mean"), confirmed=(TARGET_COL, "mean")).round(3).to_string())
