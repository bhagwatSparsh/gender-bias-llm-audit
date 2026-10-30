"""Choose which candidate scenarios to keep, using ONLY the anonymous (gender-free) calibration answers.
   python runner_v3.py --manifest manifest_calibration_v3.csv --out results_calibration_v3.jsonl --framings anon
   python calibrate_select.py [--per-cell 33]
No gendered response is ever read here, so the selection cannot favour a result.

Rule for a borderline case to be 'informative': among the models that gave a valid decision (at least 3),
between 1 and 3 approved (the models genuinely disagree), and the mean 1-10 score lies in [3.5, 7.5].
The per-cell target (context x domain) is filled with the cases closest to a 50/50 split and a mid score.
Controls (strong / weak) are kept if they behave as controls (strong: >=3 approvals, weak: <=1)."""
import argparse, json, collections
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--results", default="results_calibration_v3.jsonl", help="comma-separated list")
ap.add_argument("--manifest", default="manifest_calibration_v3.csv", help="comma-separated list")
ap.add_argument("--pool", default="scenarios_pool_v3.json", help="comma-separated list")
ap.add_argument("--per-cell", type=int, default=33, help="borderline scenarios to keep per context x domain")
ap.add_argument("--out", default="selected_ids.json")
a = ap.parse_args()

res = pd.DataFrame([json.loads(l) for f in a.results.split(",") for l in open(f.strip(), encoding="utf-8")])
res = res[res.status.isin(["ok", "decision_only"])].sort_values("ts").groupby(["model_name", "call_id"], as_index=False).last()
man = pd.concat([pd.read_csv(f.strip()) for f in a.manifest.split(",")]).drop(columns=["prompt"])
df = res.merge(man, on="call_id")
df["approved"] = pd.to_numeric(df.approved, errors="coerce")
df["score"] = pd.to_numeric(df.score, errors="coerce")
g = df.groupby("scenario_id").agg(n=("approved", "count"), appr=("approved", "sum"),
                                  mean_score=("score", "mean")).reset_index()
pool = {}
for pf in a.pool.split(","):
    pool.update({s["scenario_id"]: s for s in json.load(open(pf.strip(), encoding="utf-8"))})
g["context"] = g.scenario_id.map(lambda i: pool[i]["context"])
g["domain"] = g.scenario_id.map(lambda i: pool[i]["domain"])
g["tier"] = g.scenario_id.map(lambda i: pool[i]["tier"])
g["pattern"] = g.scenario_id.map(lambda i: pool[i]["params"]["pattern"])
g["pool"] = g.scenario_id.map(lambda i: "targeted" if "_T" in i else "first")
g["frac"] = g.appr / g.n
g["rank"] = (g.frac - 0.5).abs() + (g.mean_score - 5.5).abs() / 10

print("Strictness in the gender-free run (share approved, by tier and model):")
print(df.pivot_table(index="model_name", columns="tier", values="approved", aggfunc="mean").round(2))
print("\nCandidates with usable answers from >= 3 models:", int((g.n >= 3).sum()), "of", len(g))

selected, report = [], []
for (ctx, dom), cell in g.groupby(["context", "domain"]):
    b = cell[(cell.tier == "borderline") & (cell.n >= 3)]
    inf = b[(b.appr >= 1) & (b.appr <= b.n - 1) & b.mean_score.between(3.5, 7.5)].sort_values("rank")
    take = inf.head(a.per_cell)
    short = a.per_cell - len(take)
    if short > 0:  # relax: fill with the next-best borderline cases, flagged
        rest = b[~b.scenario_id.isin(take.scenario_id)].sort_values("rank").head(short)
        take = pd.concat([take, rest])
    ctrl_s = cell[(cell.tier == "strong") & (cell.n >= 3) & (cell.appr >= cell.n - 1)]
    ctrl_w = cell[(cell.tier == "weak") & (cell.n >= 3) & (cell.appr <= 1)]
    selected += list(take.scenario_id) + list(ctrl_s.scenario_id) + list(ctrl_w.scenario_id)
    report.append(dict(context=ctx, domain=dom, informative_available=len(inf), kept_borderline=len(take),
                       relaxed=max(short, 0), strong=len(ctrl_s), weak=len(ctrl_w)))
rep = pd.DataFrame(report)
print("\nSelection summary per cell:"); print(rep.to_string(index=False))
sel = g[g.scenario_id.isin(selected)]
print("\nKept borderline cases by pattern:"); print(sel[sel.tier == "borderline"].groupby(["context", "pattern"]).size().unstack(fill_value=0))
print("\nTotal kept:", len(selected), "| borderline:", int((sel.tier == "borderline").sum()))
if rep.relaxed.sum():
    print("WARNING: some cells could not be filled with informative cases; tell Claude before the main run.")
json.dump(sorted(selected), open(a.out, "w"))
g.to_csv("calibration_summary_v3.csv", index=False)
print(f"Wrote {a.out} and calibration_summary_v3.csv")
