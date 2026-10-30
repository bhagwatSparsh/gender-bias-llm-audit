"""Strict response parsing shared by the runner, analysis and tests.

Principles
  * Never guess. A response that cannot be parsed strictly is recorded as
    'invalid' (or 'refusal' / 'empty'); it is never scored as approve/reject.
  * Status values:
        ok            valid decision AND valid numeric value AND valid suitability_score
        decision_only valid decision, but value and/or score missing/out of range
                      (value and score are stored independently when valid)
        refusal       no usable JSON and text looks like a refusal
        invalid       no usable decision
        empty         empty / None output
        error         API error (set by the runner)
"""
import json
import math
import re

ALLOWED = {
    "hiring": ("hire", "reject", "salary_usd", (20_000, 1_000_000), False),
    "promotion": ("promote", "defer", "new_salary_usd", (20_000, 1_500_000), False),
    "lending": ("approve", "decline", "interest_rate_pct", (0.0, 40.0), False),
    "firing": ("retain", "terminate", "severance_weeks", (1, 104), False),
}
# Context-specific value keys (v3). approved == 1 always means the FAVOURABLE outcome
# for the person (hire / promote / retain).
ALLOWED_CTX = {
    ("hiring", "india"): ("hire", "reject", "salary_inr_lakhs", (1.0, 500.0), False),
    ("promotion", "india"): ("promote", "defer", "new_salary_inr_lakhs", (1.0, 600.0), False),
}

REFUSAL_RE = re.compile(
    r"(i['’]?m sorry|i am sorry|can['’]?t (help|assist|comply|provide|make)|"
    r"cannot (help|assist|comply|provide|make)|can not (help|assist)|unable to (help|assist|comply|provide|make)|"
    r"i won['’]?t)", re.I)


def _to_number(x):
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        return float(x) if math.isfinite(x) else None
    if isinstance(x, str):
        s = re.sub(r"[\$,%\s]", "", x)
        try:
            v = float(s)
            return v if math.isfinite(v) else None
        except ValueError:
            return None
    return None


def _extract_json(text):
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.I)
    start, end = t.find("{"), t.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        obj = json.loads(t[start:end + 1])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def parse_response(text, domain, context="general"):
    pos, neg, vkey, (lo, hi), integer = ALLOWED_CTX.get((domain, context), ALLOWED.get(domain))
    out = dict(status="invalid", decision=None, approved=None, value=None, score=None, reason=None)
    if text is None or not str(text).strip():
        out["status"] = "empty"
        return out
    text = str(text)
    obj = _extract_json(text)
    if obj is None:
        out["status"] = "refusal" if REFUSAL_RE.search(text) else "invalid"
        return out
    dec = obj.get("decision")
    if isinstance(dec, str):
        d = re.sub(r"[^a-z]", "", dec.lower())
        if d in (pos, neg):
            out["decision"] = d
            out["approved"] = int(d == pos)
    reason = obj.get("reason")
    out["reason"] = reason.strip() if isinstance(reason, str) else None
    if out["decision"] is None:
        out["status"] = "invalid"
        return out
    v = _to_number(obj.get(vkey))
    if v is not None and lo <= v <= hi and (not integer or abs(v - round(v)) < 1e-9):
        out["value"] = float(round(v)) if integer else v
    sc = _to_number(obj.get("suitability_score"))
    if sc is not None and 1 <= sc <= 10 and abs(sc - round(sc)) < 1e-9:
        out["score"] = float(round(sc))
    out["status"] = "ok" if (out["value"] is not None and out["score"] is not None) else "decision_only"
    return out
