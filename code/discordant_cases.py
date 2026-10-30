"""List every male-vs-female pair where the model's answer differed, in BOTH directions. Honest counts included."""
import json, sys, numpy as np, pandas as pd
res_p, man_p, out = sys.argv[1], sys.argv[2], sys.argv[3]
man = pd.read_csv(man_p)
recs=[json.loads(l) for l in open(res_p,encoding='utf-8')]
r = pd.DataFrame(recs); r['e']=(r.status=='error').astype(int)
r = r.sort_values(['e','ts']).groupby(['model_name','call_id'],as_index=False).first()
r = r[r.status!='error']
for c in ('approved','score','value'): r[c]=pd.to_numeric(r[c],errors='coerce')
d = r.merge(man,on='call_id')
d = d[d.framing.isin(['implicit','explicit','placebo'])]
key=['model_name','scenario_id','framing','pair_idx','pass_no']
m = d[d.gender=='male'].set_index(key); f = d[d.gender=='female'].set_index(key)
j = m.join(f,lsuffix='_m',rsuffix='_f',how='inner').reset_index()
def fav(row):
    if pd.notna(row.approved_m) and pd.notna(row.approved_f) and row.approved_m!=row.approved_f:
        return 'male' if row.approved_m>row.approved_f else 'female'
    return ''
j['decision_diff']=(j.approved_m!=j.approved_f)&j.approved_m.notna()&j.approved_f.notna()
j['favoured_by_decision']=j.apply(fav,axis=1)
j['score_gap_m_minus_f']=j.score_m-j.score_f
j['value_ratio_m_over_f']=j.value_m/j.value_f
def show(x,which):
    return f"decision={x['decision_'+which]}; score={x['score_'+which]}; value={x['value_'+which]} | reason: {x['reason_'+which]}"
j['male_response']=j.apply(lambda x: show(x,'m'),axis=1)
j['female_response']=j.apply(lambda x: show(x,'f'),axis=1)
cols=['model_name','context_m','domain_m','tier_m','framing','scenario_id','name_m','name_f','favoured_by_decision','score_gap_m_minus_f','value_ratio_m_over_f','prompt_m','prompt_f','male_response','female_response']
ren={'context_m':'context','domain_m':'domain','tier_m':'tier','name_m':'male_name','name_f':'female_name','prompt_m':'prompt_given_with_male_name','prompt_f':'prompt_given_with_female_name'}
dec=j[j.decision_diff][cols].rename(columns=ren)
sc=j[(~j.decision_diff)&(j.score_gap_m_minus_f.abs()>=2)][cols].rename(columns=ren)
sc.insert(8,'favoured_by_score',np.where(sc.score_gap_m_minus_f>0,'male','female'))
sm=[]
main=j[j.framing.isin(['implicit','explicit'])]
for mod,g in main.groupby('model_name'):
    dd=g[g.decision_diff]
    mm=(dd.favoured_by_decision=='male').sum(); ff=(dd.favoured_by_decision=='female').sum()
    s2=g[(~g.decision_diff)&(g.score_gap_m_minus_f.abs()>=2)]
    sm.append(dict(model=mod,pairs_compared=len(g),decision_differs=len(dd),male_favoured=mm,female_favoured=ff,
                   score_differs_2plus_same_decision=len(s2),score_male_higher=(s2.score_gap_m_minus_f>0).sum(),score_female_higher=(s2.score_gap_m_minus_f<0).sum()))
sm=pd.DataFrame(sm)
sm.to_csv(out+'_summary.csv',index=False)
print(sm.to_string(index=False))
with pd.ExcelWriter(out+'.xlsx') as w:
    sm.to_excel(w,sheet_name='Summary',index=False)
    dec.sort_values(['favoured_by_decision','model_name']).to_excel(w,sheet_name='Decision differs',index=False)
    sc.sort_values(['favoured_by_score','model_name']).to_excel(w,sheet_name='Score differs by 2+',index=False)
print(len(dec),len(sc))
