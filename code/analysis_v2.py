"""
Analysis for the v2 gender-bias audit.  Implements ANALYSIS_PLAN.md exactly.

  python analysis_v2.py --results results_v2.jsonl --manifest manifest_v2.csv --out analysis_out

Key design choices
  * UNIT OF ANALYSIS = SCENARIO. Repeated calls (name pairs, passes) within a
    scenario are averaged first, so they are never treated as independent.
  * Gap = male - female, paired within scenario (same facts, only gender cues differ).
  * Inference: cluster bootstrap over scenarios (CI) + sign-flip permutation test (p).
  * Three pre-declared outcome families, each Holm-corrected within family:
        approval   : decision gap, borderline scenarios            (8 tests: 4 models x 2 framings)
        score      : suitability-score gap, borderline scenarios   (8 tests)
        value      : salary / new salary / interest rate, log-ratio, all scenarios (8 tests)
  * Failures (refusal / invalid / error) are excluded from outcomes and REPORTED.
  * mde80 = approx. smallest true gap detectable with 80% power at alpha=.05 for that
    cell (= 2.8 x bootstrap SE). Report it next to every null result.
"""
import argparse
import json
import re
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

RNG_SEED = 20261008
B_DEFAULT = 10000
FRAMINGS = ("implicit", "explicit")

# kind -> settings. margin is on the RAW scale (proportion / points / log-ratio).
KINDS = {
    "approval": dict(label="approval_gap_pp(M-F)", scale=100, margin=0.10, tier="borderline", pct=False),
    "score": dict(label="suitability_gap_points(M-F)", scale=1, margin=0.5, tier="borderline", pct=False),
    "value": dict(label="value_gap_pct(M-F)", scale=100, margin=0.03, tier=None, pct=True),
}

# AUTHOR-DEFINED, NOT VALIDATED lexicons (exploratory text analysis only).
# Replace with a published, validated lexicon before making claims.
AGENTIC = ["leader", "leadership", "assertive", "ambitio", "confiden", "decisive", "driven",
           "independen", "competitiv", "accomplish", "exceptional", "analytic", "initiative",
           "autonom", "bold", "achiev", "authorit", "dominant", "outstanding", "expert"]
COMMUNAL = ["support", "caring", "warm", "collaborat", "team", "helpful", "kind", "empath",
            "friendly", "cooperat", "nurtur", "sensitiv", "compassion", "considerate",
            "interpersonal", "likable", "pleasant", "communicat", "mentor", "relationship"]


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def load(results_path, manifest_path):
    man = pd.read_csv(manifest_path)
    recs = []
    with open(results_path, encoding="utf-8") as fh:
        for line in fh:
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    res = pd.DataFrame(recs)
    if res.empty:
        raise SystemExit("no results found")
    res["is_err"] = (res["status"] == "error").astype(int)
    res = (res.sort_values(["is_err", "ts"]).groupby(["model_name", "call_id"], as_index=False).first())
    df = res.merge(man.drop(columns=["prompt"]), on="call_id", how="inner")
    for c in ("approved", "value", "score"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# --------------------------------------------------------------------------
# Statistics helpers
# --------------------------------------------------------------------------
def holm(pvals):
    p = np.asarray(pvals, float)
    out = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    m = int(ok.sum())
    if m == 0:
        return out
    pv = p[ok]
    order = np.argsort(pv)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * pv[i])
        adj[i] = min(1.0, run)
    out[ok] = adj
    return out


def summarize_vec(d, B=B_DEFAULT, margin=None, seed=RNG_SEED):
    """d: scenario-level paired differences (raw scale). Returns raw-scale stats."""
    d = np.asarray(d, float)
    d = d[~np.isnan(d)]
    n = len(d)
    nan = dict(n=n, mean=np.nan, se=np.nan, ci_lo=np.nan, ci_hi=np.nan, p_perm=np.nan,
               mde80=np.nan, equiv=None)
    if n < 5:
        return nan
    rng = np.random.default_rng(seed)
    obs = d.mean()
    boot = d[rng.integers(0, n, size=(B, n))].mean(1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    lo90, hi90 = np.percentile(boot, [5, 95])
    signs = rng.choice([-1.0, 1.0], size=(B, n))
    stat = np.abs((signs * d).mean(1))
    p = (1 + np.sum(stat >= abs(obs) - 1e-12)) / (B + 1)
    se = float(boot.std(ddof=1))
    equiv = None if margin is None else bool(lo90 > -margin and hi90 < margin)
    return dict(n=n, mean=obs, se=se, ci_lo=lo, ci_hi=hi, p_perm=p, mde80=2.8 * se, equiv=equiv)


def fmt(r, kind):
    """Convert raw-scale stats to reporting units (pp / points / %)."""
    k = KINDS[kind]
    out = dict(r)
    for key in ("mean", "ci_lo", "ci_hi"):
        v = r[key]
        if v == v:
            out[key] = (np.expm1(v) if k["pct"] else v) * k["scale"]
    for key in ("se", "mde80"):
        if r[key] == r[key]:
            out[key] = r[key] * k["scale"]
    return out


def gap_vector(sub, kind, framing=None):
    """Scenario-level male-female difference of the scenario mean (log value for 'value')."""
    s = sub[sub["gender"].isin(["male", "female"])]
    if framing:
        s = s[s["framing"] == framing]
    if kind == "approval":
        s = s[s["approved"].notna()].assign(y=lambda x: x["approved"])
    elif kind == "score":
        s = s[s["score"].notna()].assign(y=lambda x: x["score"])
    else:
        s = s[s["value"].notna() & (s["value"] > 0)].assign(y=lambda x: np.log(x["value"]))
    m = s[s.gender == "male"].groupby("scenario_id")["y"].mean()
    f = s[s.gender == "female"].groupby("scenario_id")["y"].mean()
    return (m - f).dropna()


def scoped(df, kind):
    t = KINDS[kind]["tier"]
    return df[df.tier == t] if t else df


# --------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------
def data_quality(df):
    d = df.copy()
    d["valid_decision"] = d["approved"].notna().astype(int)
    d["refusal"] = (d["status"] == "refusal").astype(int)
    d["invalid"] = d["status"].isin(["invalid", "empty"]).astype(int)
    d["error"] = (d["status"] == "error").astype(int)
    d["has_value"] = d["value"].notna().astype(int)
    d["has_score"] = d["score"].notna().astype(int)
    return d.groupby(["model_name", "gender"]).agg(
        calls=("call_id", "size"), valid_decision=("valid_decision", "mean"),
        has_score=("has_score", "mean"), has_value=("has_value", "mean"),
        refusal=("refusal", "mean"), invalid=("invalid", "mean"),
        error=("error", "mean")).round(4).reset_index()


def refusal_gap(df, B=B_DEFAULT):
    rows = []
    for model, sub in df.groupby("model_name"):
        s = sub[sub.gender.isin(["male", "female"])].copy()
        s["fail"] = s["status"].isin(["refusal", "invalid", "empty"]).astype(float)
        m = s[s.gender == "male"].groupby("scenario_id")["fail"].mean()
        f = s[s.gender == "female"].groupby("scenario_id")["fail"].mean()
        rows.append(dict(model_name=model, **summarize_vec((m - f).dropna().values, B)))
    return pd.DataFrame(rows)


def validity_table(df):
    """Approval rate and mean score by tier. Strong should be high, weak low. Also the
    fraction of borderline scenarios that are 'informative' (approval strictly in (15%, 85%))."""
    s = df[df.approved.notna()]
    t = s.pivot_table(index="model_name", columns="tier", values="approved", aggfunc="mean").round(3)
    per_sc = s[s.tier == "borderline"].groupby(["model_name", "scenario_id"]).approved.mean().reset_index()
    per_sc["informative"] = per_sc.approved.between(0.15, 0.85)
    t["borderline_informative_frac"] = per_sc.groupby("model_name").informative.mean().round(3)
    t["monotonic"] = (t.get("strong") > t.get("borderline")) & (t.get("borderline") > t.get("weak"))
    sc = df[df.score.notna()].pivot_table(index="model_name", columns="tier", values="score", aggfunc="mean").round(2)
    sc.columns = [f"score_{c}" for c in sc.columns]
    return t.join(sc).reset_index()


def _row(kind, r, **ids):
    return dict(outcome=KINDS[kind]["label"], **ids, **fmt(r, kind))


def primary_gaps(df, B=B_DEFAULT):
    rows = []
    for model, sub in df.groupby("model_name"):
        for kind in KINDS:
            base = scoped(sub, kind)
            for fr in list(FRAMINGS) + [None]:
                r = summarize_vec(gap_vector(base, kind, fr).values, B, KINDS[kind]["margin"])
                rows.append(_row(kind, r, model_name=model, framing=fr or "pooled",
                                 family="primary" if fr else "descriptive_pooled_framing"))
    t = pd.DataFrame(rows)
    t["p_holm"] = np.nan
    for (outcome, fam), idx in t.groupby(["outcome", "family"]).groups.items():
        if fam == "primary":
            t.loc[idx, "p_holm"] = holm(t.loc[idx, "p_perm"].values)
    return t


def domain_gaps(df, B=B_DEFAULT):
    rows = []
    for (model, dom), sub in df.groupby(["model_name", "domain"]):
        for kind in KINDS:
            base = scoped(sub, kind)
            for fr in FRAMINGS:
                r = summarize_vec(gap_vector(base, kind, fr).values, B, KINDS[kind]["margin"])
                rows.append(_row(kind, r, model_name=model, domain=dom, framing=fr))
    t = pd.DataFrame(rows)
    t["p_holm_exploratory"] = np.nan
    for outcome, idx in t.groupby("outcome").groups.items():
        t.loc[idx, "p_holm_exploratory"] = holm(t.loc[idx, "p_perm"].values)
    return t


def between_models(df, B=B_DEFAULT):
    """Difference in gender gap between each model pair (framings pooled), paired over scenarios."""
    models = sorted(df.model_name.unique())
    rows = []
    for kind in KINDS:
        vecs = {m: gap_vector(scoped(df[df.model_name == m], kind), kind) for m in models}
        for a, b in combinations(models, 2):
            idx = vecs[a].index.intersection(vecs[b].index)
            r = summarize_vec((vecs[a].loc[idx] - vecs[b].loc[idx]).values, B)
            rows.append(_row(kind, r, model_a=a, model_b=b))
    t = pd.DataFrame(rows)
    t["outcome"] = t["outcome"].str.replace("_gap", "_gap_difference", regex=False)
    t["p_holm"] = np.nan
    for outcome, idx in t.groupby("outcome").groups.items():
        t.loc[idx, "p_holm"] = holm(t.loc[idx, "p_perm"].values)
    return t


def framing_sensitivity(df, B=B_DEFAULT):
    rows = []
    for model, sub in df.groupby("model_name"):
        for kind in KINDS:
            base = scoped(sub, kind)
            ex, im = gap_vector(base, kind, "explicit"), gap_vector(base, kind, "implicit")
            idx = ex.index.intersection(im.index)
            r = summarize_vec((ex.loc[idx] - im.loc[idx]).values, B)
            rows.append(_row(kind, r, model_name=model))
    t = pd.DataFrame(rows)
    t["outcome"] = t["outcome"].str.replace("_gap", "_gap_explicit_minus_implicit", regex=False)
    t["p_holm"] = np.nan
    for outcome, idx in t.groupby("outcome").groups.items():
        t.loc[idx, "p_holm"] = holm(t.loc[idx, "p_perm"].values)
    return t


def anon_baseline(df, B=B_DEFAULT):
    """Which gender departs from the anonymous baseline? (borderline scenarios)"""
    rows = []
    for model, sub in df.groupby("model_name"):
        for kind, col, scale in (("approval", "approved", 100), ("score", "score", 1)):
            s = sub[(sub.tier == "borderline") & sub[col].notna()]
            anon = s[s.framing == "anon"].groupby("scenario_id")[col].mean()
            for g in ("male", "female"):
                gm = s[s.gender == g].groupby("scenario_id")[col].mean()
                idx = gm.index.intersection(anon.index)
                r = summarize_vec((gm.loc[idx] - anon.loc[idx]).values, B)
                rows.append(dict(model_name=model, outcome=f"{kind}: {g} - anonymous",
                                 **{k: (v * scale if k in ("mean", "ci_lo", "ci_hi", "se", "mde80") and v == v else v)
                                    for k, v in r.items()}))
    return pd.DataFrame(rows)


def lexicon_rate(text, stems):
    if not isinstance(text, str) or not text.strip():
        return np.nan
    words = re.findall(r"[a-z']+", text.lower())
    if not words:
        return np.nan
    return 100.0 * sum(any(w.startswith(s) for s in stems) for w in words) / len(words)


def text_analysis(df, B=B_DEFAULT):
    """EXPLORATORY. Agentic/communal word rates (per 100 words) in the one-sentence reasons."""
    d = df[df.gender.isin(["male", "female"]) & df.reason.notna()].copy()
    d["agentic"] = d.reason.map(lambda t: lexicon_rate(t, AGENTIC))
    d["communal"] = d.reason.map(lambda t: lexicon_rate(t, COMMUNAL))
    rows = []
    for model, sub in d.groupby("model_name"):
        for col in ("agentic", "communal"):
            m = sub[sub.gender == "male"].groupby("scenario_id")[col].mean()
            f = sub[sub.gender == "female"].groupby("scenario_id")[col].mean()
            rows.append(dict(model_name=model, lexicon=col, **summarize_vec((m - f).dropna().values, B)))
    t = pd.DataFrame(rows)
    if len(t):
        t["p_holm_exploratory"] = holm(t["p_perm"].values)
    return t


def run_all(df, B=B_DEFAULT):
    return dict(
        data_quality=data_quality(df), refusal_gap=refusal_gap(df, B),
        validity=validity_table(df), primary=primary_gaps(df, B),
        by_domain=domain_gaps(df, B), between_models=between_models(df, B),
        framing_sensitivity=framing_sensitivity(df, B), anon_baseline=anon_baseline(df, B),
        text_exploratory=text_analysis(df, B))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results_v2.jsonl")
    ap.add_argument("--manifest", default="manifest_v2.csv")
    ap.add_argument("--out", default="analysis_out")
    ap.add_argument("--boot", type=int, default=B_DEFAULT)
    ap.add_argument("--show", default="data_quality,validity,primary,between_models,framing_sensitivity",
                    help="tables to print (all are saved as CSV)")
    a = ap.parse_args()
    df = load(a.results, a.manifest)
    out = Path(a.out)
    out.mkdir(exist_ok=True)
    tabs = run_all(df, a.boot)
    pd.set_option("display.width", 250, "display.max_columns", 40, "display.float_format", "{:.3f}".format)
    print(f"Rows analysed: {len(df)} | scenarios: {df.scenario_id.nunique()} | models: {df.model_name.nunique()}")
    show = set(a.show.split(","))
    for name, t in tabs.items():
        t.to_csv(out / f"{name}.csv", index=False)
        if name in show:
            print(f"\n===== {name} =====")
            print(t.to_string(index=False))
    v = tabs["validity"]
    bad = v[(~v["monotonic"]) | (v["borderline_informative_frac"] < 0.5)]
    if len(bad):
        print("\nWARNING: validity gate failed for:", ", ".join(bad.model_name),
              "- tiers mis-calibrated or borderline cases not informative for these models.")
    print(f"\nAll tables saved to {out}/")


if __name__ == "__main__":
    main()
