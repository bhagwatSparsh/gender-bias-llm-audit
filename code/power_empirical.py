import sys; sys.path.insert(0,'/home/claude/gender_bias_v2')
import numpy as np, pandas as pd
from itertools import combinations
from scipy.stats import norm
import analysis_v2 as A
df = A.load('/root/.claude/uploads/8bd3010c-1c9b-508a-8157-6e047ee31ff5/67bd688e-results_today.jsonl','today/manifest_v2.csv')
models = sorted(df.model_name.unique())
out=[]
for kind in ("score","approval"):
    k=[x for x in A.KINDS if kind in x][0]
    vec={m:A.gap_vector(A.scoped(df[df.model_name==m],k),k) for m in models}
    sd_single={m:float(vec[m].std(ddof=1)) for m in models}
    print(kind,"per-scenario SD of gap:",{m:round(v,3) for m,v in sd_single.items()})
    sds=[]
    for a,b in combinations(models,2):
        idx=vec[a].index.intersection(vec[b].index); d=(vec[a].loc[idx]-vec[b].loc[idx])
        sds.append(float(d.std(ddof=1)))
    print(kind,"per-scenario SD of between-model difference: median %.3f, max %.3f"%(np.median(sds),max(sds)))
    if kind=="score":
        za=norm.ppf(1-0.05/12); zb=norm.ppf(0.8)   # two-sided alpha 0.05/6 pairs (Bonferroni, conservative)
        for delta in (0.3,0.2,0.15,0.1):
            n=lambda s:int(np.ceil(((za+zb)*s/delta)**2))
            print(f" delta={delta}: n borderline scenarios, median-pair sd -> {n(np.median(sds))}, worst-pair sd -> {n(max(sds))}")
        # also: single-model gap detection
        for m in models:
            for delta in (0.3,0.2,0.1):
                z=norm.ppf(1-0.05/16)+zb
                print(f" single {m} delta={delta}: n={int(np.ceil((z*sd_single[m]/delta)**2))}")
