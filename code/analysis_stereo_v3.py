"""Analysis for the stereotype test (Study 3). Three PRE-DECLARED hypotheses x 4 models = 12 primary tests,
Holm-corrected together. See PREREGISTRATION_STEREO.md (written before any real run).

  python analysis_stereo_v3.py --results results_stereo_v3.jsonl --manifest manifest_stereo_v3.csv --out stereo_out

H1 pronoun  Stereotype use: among sentences the model resolves (not 'unclear'), the share resolved in the
            stereotype-consistent way (he -> male-stereotyped job, she -> female-stereotyped job) is above 50%.
            Statistic per item: consistency - 0.5.  Positive = stereotype used.
H2 letter   Same facts, male vs female name: (agentic-word rate - communal-word rate) is higher for male names.
            Statistic per occupation: male - female.  Positive = male-leaning wording.
H3 role     Same qualifications, male vs female name: share of recommendations to the MALE-stereotyped role is higher for
            male names.  Statistic per profile: P(male role | male name) - P(male role | female name).  Positive = male-leaning.

Unit of analysis = item (sentence pair / occupation / profile); repeats and counterbalanced orders are averaged inside it.
Inference = cluster bootstrap over items + sign-flip permutation test, as in the main study. mde80 = 2.8 x SE.
The word lists for H2 are author-defined and NOT validated (exploratory-grade); the paper says so.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis_v3 import AGENTIC, COMMUNAL, B_DEFAULT, holm, lexicon_rate, summarize_vec

MODEL_ORDER = ["GPT-OSS-120B", "GPT-OSS-20B", "Qwen-3.8-27B", "Gemini-3.5-Flash-Lite"]


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
    res = res.sort_values(["is_err", "ts"]).groupby(["model_name", "call_id"], as_index=False).first()
    res = res[res.status != "error"]
    df = res.merge(man.drop(columns=["prompt"]), on="call_id", how="inner")
    return df, man


def row(model, hyp, label, vec, B, extra=None):
    r = summarize_vec(np.asarray(vec, float), B)
    out = dict(model_name=model, hypothesis=hyp, quantity=label, **r)
    out.update(extra or {})
    return out


def h1_items(d):
    """per item: consistency among resolved, plus unclear rate and he/she split"""
    d = d[d.domain == "pronoun"].copy()
    d["unclear"] = (d.decision == "unclear").astype(float)
    r = d[d.decision.isin(["first", "second"])].copy()
    male_pron = r.gender == "male"
    first_male = r.first_is_male_occ.astype(float) == 1
    ref_is_male_occ = (r.decision == "first") == first_male
    r["consistent"] = np.where(male_pron, ref_is_male_occ, ~ref_is_male_occ).astype(float)
    r["ref_male_occ"] = ref_is_male_occ.astype(float)
    return d, r


def analyse(df, B):
    rows, desc = [], []
    for model in MODEL_ORDER:
        d = df[df.model_name == model]
        if d.empty:
            continue
        # ---- H1 pronoun
        allp, res = h1_items(d)
        if len(res):
            c = res.groupby("item_id").consistent.mean()
            rows.append(row(model, "H1", "stereotype-consistent share - 0.5", (c - 0.5).values, B))
            he = res[res.gender == "male"].groupby("item_id").consistent.mean()
            she = res[res.gender == "female"].groupby("item_id").consistent.mean()
            rows.append(row(model, "H1-secondary", "consistency(he) - consistency(she)", (he - she).dropna().values, B))
            m_he = res[res.gender == "male"].groupby("item_id").ref_male_occ.mean()
            m_she = res[res.gender == "female"].groupby("item_id").ref_male_occ.mean()
            rows.append(row(model, "H1-secondary", "P(resolves to male job | he) - P(... | she)", (m_he - m_she).dropna().values, B))
            for ctx, g in res.groupby("context"):
                cc = g.groupby("item_id").consistent.mean()
                rows.append(row(model, "H1-by-context", f"{ctx}: consistent share - 0.5", (cc - 0.5).values, B))
            desc.append(dict(model_name=model, task="pronoun", measure="share answered 'unclear'",
                             value=allp.unclear.mean(), n=len(allp)))
            desc.append(dict(model_name=model, task="pronoun", measure="stereotype-consistent share (resolved)",
                             value=res.consistent.mean(), n=len(res)))
        # ---- H3 role
        r = d[(d.domain == "role") & d.decision.isin(["X", "Y"])].copy()
        if len(r):
            x_male = r.role_x_is_male.astype(float) == 1
            r["male_role"] = ((r.decision == "X") == x_male).astype(float)
            m = r[r.gender == "male"].groupby("item_id").male_role.mean()
            f = r[r.gender == "female"].groupby("item_id").male_role.mean()
            rows.append(row(model, "H3", "P(male role | male name) - P(male role | female name)", (m - f).dropna().values, B))
            for ctx, g in r.groupby("context"):
                mm = g[g.gender == "male"].groupby("item_id").male_role.mean()
                ff = g[g.gender == "female"].groupby("item_id").male_role.mean()
                rows.append(row(model, "H3-by-context", f"{ctx}: P(male role) male - female", (mm - ff).dropna().values, B))
            desc.append(dict(model_name=model, task="role", measure="share recommending the male-stereotyped role (all)",
                             value=r.male_role.mean(), n=len(r)))
        # ---- H2 letter
        l = d[(d.domain == "letter") & (d.status == "ok")].copy()
        if len(l):
            l["agentic"] = l.reason.map(lambda t: lexicon_rate(t, AGENTIC))
            l["communal"] = l.reason.map(lambda t: lexicon_rate(t, COMMUNAL))
            l["index"] = l.agentic - l.communal
            l["words"] = l.reason.str.split().str.len()
            for col, label, hyp in (("index", "agentic rate - communal rate (per 100 words), male - female", "H2"),
                                    ("agentic", "agentic rate, male - female", "H2-secondary"),
                                    ("communal", "communal rate, male - female", "H2-secondary"),
                                    ("words", "letter length in words, male - female", "H2-secondary")):
                m = l[l.gender == "male"].groupby("item_id")[col].mean()
                f = l[l.gender == "female"].groupby("item_id")[col].mean()
                rows.append(row(model, hyp, label, (m - f).dropna().values, B))
            desc.append(dict(model_name=model, task="letter", measure="mean agentic rate per 100 words",
                             value=l.agentic.mean(), n=len(l)))
            desc.append(dict(model_name=model, task="letter", measure="mean communal rate per 100 words",
                             value=l.communal.mean(), n=len(l)))
    t = pd.DataFrame(rows)
    t["p_holm"] = np.nan
    prim = t.hypothesis.isin(["H1", "H2", "H3"])
    t.loc[prim, "p_holm"] = holm(t.loc[prim, "p_perm"].values)
    t["primary"] = prim
    return t, pd.DataFrame(desc)


def quality(df, man):
    d = df.copy()
    d["ok"] = (d.status == "ok").astype(int)
    q = d.groupby(["model_name", "domain"]).agg(answers=("call_id", "size"), readable=("ok", "mean")).reset_index()
    exp = man.groupby("domain").size().rename("expected_per_model")
    return q.merge(exp, left_on="domain", right_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results_stereo_v3.jsonl")
    ap.add_argument("--manifest", default="manifest_stereo_v3.csv")
    ap.add_argument("--out", default="stereo_out")
    ap.add_argument("--boot", type=int, default=B_DEFAULT)
    a = ap.parse_args()
    df, man = load(a.results, a.manifest)
    out = Path(a.out)
    out.mkdir(exist_ok=True)
    pd.set_option("display.width", 250, "display.max_columns", 40, "display.float_format", "{:.3f}".format,
                  "display.max_colwidth", 70)
    q = quality(df, man)
    q.to_csv(out / "data_quality.csv", index=False)
    print("===== data quality =====")
    print(q.to_string(index=False))
    t, desc = analyse(df, a.boot)
    t.to_csv(out / "results.csv", index=False)
    desc.to_csv(out / "descriptives.csv", index=False)
    cols = ["model_name", "hypothesis", "quantity", "n", "mean", "ci_lo", "ci_hi", "p_perm", "p_holm", "mde80"]
    print("\n===== PRIMARY (12 pre-declared tests, Holm-corrected together) =====")
    print(t[t.primary][cols].to_string(index=False))
    print("\n===== secondary / by context (descriptive; no multiplicity claim) =====")
    print(t[~t.primary][cols[:-2] + ["mde80"]].to_string(index=False))
    print("\n===== descriptives =====")
    print(desc.to_string(index=False))
    print(f"\nAll tables saved to {out}/")


if __name__ == "__main__":
    main()
