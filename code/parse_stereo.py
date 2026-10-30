"""Strict parsing for the stereotype test (Study 3). Never guesses.

Returns the same fields the main runner stores:
  status    ok | invalid | refusal | empty
  decision  pronoun: first / second / unclear ; role: X / Y ; letter: 'letter'
  approved  None (not used)
  value     None
  score     role: confidence 1-10 (else None)
  reason    pronoun/role: one-sentence reason ; letter: the full letter text
"""
import json
import re

from parse_utils import REFUSAL_RE, _extract_json


def parse_stereo(text, task):
    out = dict(status="invalid", decision=None, approved=None, value=None, score=None, reason=None)
    if text is None or not str(text).strip():
        out["status"] = "empty"
        return out
    obj = _extract_json(str(text))
    if not isinstance(obj, dict):
        out["status"] = "refusal" if REFUSAL_RE.search(str(text)) else "invalid"
        return out
    if task == "pronoun":
        r = str(obj.get("referent", "")).strip().lower()
        if r in ("first", "second", "unclear"):
            out["decision"] = r
            out["status"] = "ok"
        out["reason"] = str(obj.get("reason", ""))[:400]
    elif task == "role":
        r = str(obj.get("recommended_role", "")).strip().upper().replace("ROLE", "").strip()
        if r in ("X", "Y"):
            out["decision"] = r
            out["status"] = "ok"
        try:
            c = float(obj.get("confidence"))
            if 1 <= c <= 10 and abs(c - round(c)) < 1e-9:
                out["score"] = float(round(c))
        except (TypeError, ValueError):
            pass
        out["reason"] = str(obj.get("reason", ""))[:400]
    elif task == "letter":
        t = obj.get("letter")
        if isinstance(t, str) and len(re.findall(r"[A-Za-z']+", t)) >= 20:
            out["decision"] = "letter"
            out["reason"] = t
            out["status"] = "ok"
    return out
