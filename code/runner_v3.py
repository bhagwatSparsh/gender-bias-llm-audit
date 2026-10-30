"""
Resumable, budget-aware runner for the v2 gender-bias audit.

  python runner_v2.py --dry-run                      # show progress + ETA, no API calls
  python runner_v2.py --limit 20                     # smoke test: 20 calls per model
  python runner_v2.py                                # run until budgets are hit; re-run later to resume
  python runner_v2.py --models GPT-OSS-120B          # one model only
  python runner_v2.py --max-pass 1                   # only pass 1 (stage-1 data)
  python runner_v2.py --mock                         # offline test, no keys needed

Every response is appended to results_v2.jsonl immediately, so you can stop and
restart at any time. Failures (refusals, invalid JSON, API errors) are stored
with their status and are NEVER converted into decisions.
"""
import argparse
import collections
import csv
import hashlib
import json
import os
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from parse_utils import parse_response

MODELS = {
    "GPT-OSS-120B": dict(id="openai/gpt-oss-120b", provider="groq"),
    "Qwen-3.8-27B": dict(id="qwen/qwen3.8-27b", provider="groq"),
    "GPT-OSS-20B": dict(id="openai/gpt-oss-20b", provider="groq"),
    "Gemini-3.5-Flash-Lite": dict(id="gemini-3.5-flash-lite", provider="google"),
}


class DailyLimit(Exception):
    pass


# --------------------------------------------------------------------------
# Rate limiting
# --------------------------------------------------------------------------
class Limiter:
    """Sliding-window limiter for requests/min and tokens/min, plus a rolling
    24h budget computed from the results file (conservative vs. calendar-day
    resets)."""

    def __init__(self, rpm, tpm, daily_tokens, daily_requests, history):
        self.min_interval = 60.0 / rpm if rpm else 0.0
        self.tpm = tpm
        self.daily_tokens = daily_tokens
        self.daily_requests = daily_requests
        self.window = collections.deque()          # (ts, tokens) last 60 s
        self.day = collections.deque(history)      # (ts, tokens) last 24 h
        self.last_call = 0.0
        self.ewma_completion = 700.0               # conservative start

    def _trim(self, now):
        while self.window and now - self.window[0][0] > 60:
            self.window.popleft()
        while self.day and now - self.day[0][0] > 86400:
            self.day.popleft()

    def check_daily(self):
        now = time.time()
        self._trim(now)
        used = sum(t for _, t in self.day)
        if self.daily_tokens and used >= self.daily_tokens:
            raise DailyLimit(f"rolling-24h token budget reached ({used}/{self.daily_tokens})")
        if self.daily_requests and len(self.day) >= self.daily_requests:
            raise DailyLimit(f"rolling-24h request budget reached ({len(self.day)}/{self.daily_requests})")

    def wait(self, est_tokens):
        while True:
            now = time.time()
            self._trim(now)
            gap = self.min_interval - (now - self.last_call)
            used = sum(t for _, t in self.window)
            if gap > 0:
                time.sleep(gap)
                continue
            if self.tpm and self.window and used + est_tokens > self.tpm:
                time.sleep(max(1.0, 60 - (now - self.window[0][0]) + 0.5))
                continue
            self.last_call = time.time()
            return

    def record(self, tokens, completion):
        now = time.time()
        self.window.append((now, tokens))
        self.day.append((now, tokens))
        if completion:
            self.ewma_completion = 0.9 * self.ewma_completion + 0.1 * completion


# --------------------------------------------------------------------------
# API clients
# --------------------------------------------------------------------------
def retry_after(msg):
    m = re.search(r"try again in ([0-9.]+)\s*(ms|s|m)", msg, re.I)
    if not m:
        return None
    v = float(m.group(1))
    return v / 1000 if m.group(2).lower() == "ms" else v * 60 if m.group(2).lower() == "m" else v


def is_daily(msg):
    return bool(re.search(r"per day|tokens per day|requests per day|\bTPD\b|\bRPD\b|daily|PerDay", msg, re.I))


def make_client(provider, args):
    if args.mock:
        return None
    from dotenv import load_dotenv
    load_dotenv()
    if provider == "groq":
        from groq import Groq
        key = os.getenv("GROQ_API_KEY")
        if not key:
            raise SystemExit("GROQ_API_KEY missing in .env")
        return Groq(api_key=key)
    from google import genai
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise SystemExit("GEMINI_API_KEY missing in .env")
    return genai.Client(api_key=key)


def groq_variants(meta, args):
    """Ordered parameter variants; if the API rejects one, fall back to the next.
    The variant that worked is recorded in every result row (`extras`)."""
    if "qwen" in meta["id"]:
        v = []
        if args.qwen_effort != "default":
            v.append(dict(reasoning_effort=args.qwen_effort))
        v.append(dict(reasoning_format="hidden"))
        return v
    if "gpt-oss" in meta["id"]:
        v = []
        if args.gpt_oss_effort != "default":
            v.append(dict(include_reasoning=False, reasoning_effort=args.gpt_oss_effort))
        v.append(dict(include_reasoning=False))
        return v
    return [dict()]


def call_api(client, meta, prompt, args, state):
    """Return (text, prompt_tokens, completion_tokens, total_tokens). Raises on failure."""
    if meta["provider"] == "groq":
        variants = groq_variants(meta, args)
        i = state.get("vi", 0)
        base = dict(model=meta["id"], messages=[{"role": "user", "content": prompt}],
                    temperature=args.temperature)
        while True:
            kw = dict(base, **variants[i])
            try:
                try:
                    r = client.chat.completions.create(max_completion_tokens=args.max_completion, **kw)
                except TypeError:          # older SDK without max_completion_tokens
                    r = client.chat.completions.create(max_tokens=args.max_completion, **kw)
                state["vi"] = i
                state["extras"] = json.dumps(variants[i])
                break
            except Exception as e:  # noqa: BLE001
                msg = str(e)
                if (i + 1 < len(variants) and re.search(r"reasoning|unsupported|invalid|400", msg, re.I)
                        and not re.search(r"429|rate limit|quota", msg, re.I)):
                    i += 1
                    print(f"[{meta['id']}] API rejected {list(variants[i-1])}; falling back to {list(variants[i])}",
                          flush=True)
                    continue
                raise
        u = getattr(r, "usage", None)
        return (r.choices[0].message.content,
                getattr(u, "prompt_tokens", None), getattr(u, "completion_tokens", None),
                getattr(u, "total_tokens", None))
    from google.genai import types
    r = client.models.generate_content(
        model=meta["id"], contents=prompt,
        config=types.GenerateContentConfig(temperature=args.temperature,
                                           max_output_tokens=args.max_completion,
                                           response_mime_type="application/json"))
    state["extras"] = json.dumps({"response_mime_type": "application/json"})
    u = getattr(r, "usage_metadata", None)
    return (r.text, getattr(u, "prompt_token_count", None),
            getattr(u, "candidates_token_count", None), getattr(u, "total_token_count", None))


def mock_api(name, row, args):
    """Offline stand-in with a planted effect, for pipeline tests only."""
    h = int(hashlib.md5((name + row["call_id"]).encode()).hexdigest(), 16)
    rng = random.Random(h)
    base = {"strong": 0.97, "borderline": 0.55, "weak": 0.04}[row["tier"]]
    delta = float(os.getenv("MOCK_FEMALE_PENALTY", "0.0"))
    p = base - (delta if row["gender"] == "female" and row["tier"] == "borderline" else 0.0)
    dom, ctx = row["domain"], row.get("context", "general")
    pos = {"hiring": "hire", "promotion": "promote", "lending": "approve", "firing": "retain"}[dom]
    neg = {"hiring": "reject", "promotion": "defer", "lending": "decline", "firing": "terminate"}[dom]
    vkey = {("hiring", "general"): "salary_usd", ("promotion", "general"): "new_salary_usd",
            ("hiring", "india"): "salary_inr_lakhs", ("promotion", "india"): "new_salary_inr_lakhs",
            ("firing", "general"): "severance_weeks", ("firing", "india"): "severance_weeks",
            ("lending", "general"): "interest_rate_pct"}[(dom, ctx)]
    u = rng.random()
    if u < 0.01:
        return "I'm sorry, but I can't help with that.", 300, 12, 312
    if u < 0.02:
        return "The candidate seems fine overall.", 300, 12, 312
    dec = pos if rng.random() < p else neg
    val = {"salary_usd": lambda: int(rng.gauss(110000, 15000)), "new_salary_usd": lambda: int(rng.gauss(115000, 15000)),
           "salary_inr_lakhs": lambda: round(rng.gauss(14, 3), 1), "new_salary_inr_lakhs": lambda: round(rng.gauss(16, 3), 1),
           "severance_weeks": lambda: max(1, int(rng.gauss(12, 4))), "interest_rate_pct": lambda: round(rng.gauss(8, 1), 2)}[vkey]()
    score = max(1, min(10, int(round(rng.gauss({"strong": 9, "borderline": 5.5, "weak": 2.5}[row["tier"]], 1.2)
                                     - (float(os.getenv("MOCK_FEMALE_SCORE_PENALTY", "0"))
                                        if row["gender"] == "female" else 0)))))
    txt = json.dumps({"decision": dec, "suitability_score": score, vkey: val, "reason": "Mock reason."})
    return txt, 300, 60, 360


# --------------------------------------------------------------------------
# Worker
# --------------------------------------------------------------------------
def load_history(path, retry_errors=False):
    done, hist, ok_tokens = set(), collections.defaultdict(list), collections.defaultdict(list)
    if Path(path).exists():
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                # API errors are retried on resume; everything else is final
                if r.get("status") != "error" or not retry_errors:
                    done.add((r["model_name"], r["call_id"]))
                if r.get("tokens"):
                    hist[r["model_name"]].append((r["ts"], r["tokens"]))
    return done, hist


def worker(name, rows, args, done, hist, lock, out_fh):
    meta = MODELS[name]
    prov = meta["provider"]
    rpm = args.gemini_rpm if prov == "google" else args.groq_rpm
    tpm = None if prov == "google" else args.groq_tpm
    if args.mock:
        rpm, tpm = None, None
    dtok = None if prov == "google" else args.daily_token_budget
    dreq = args.gemini_rpd if prov == "google" else args.daily_request_budget
    lim = Limiter(rpm, tpm, dtok, dreq, hist.get(name, []) if not args.mock else [])
    client = make_client(prov, args)
    state = {}
    todo = [r for r in rows if (name, r["call_id"]) not in done and int(r["pass_no"]) <= args.max_pass
            and r["framing"] in args.framings]
    if args.limit:
        todo = todo[:args.limit]
    n_done = 0
    t0 = time.time()
    for row in todo:
        try:
            lim.check_daily()
        except DailyLimit as e:
            print(f"[{name}] STOP: {e}. Re-run later to resume.", flush=True)
            break
        est = len(row["prompt"]) / 3.2 + lim.ewma_completion
        text, pt, ct, tt, err = None, None, None, None, None
        started = time.time()
        for attempt in range(6):
            lim.wait(est)
            try:
                if args.mock:
                    text, pt, ct, tt = mock_api(name, row, args)
                else:
                    text, pt, ct, tt = call_api(client, meta, row["prompt"], args, state)
                err = None
                break
            except Exception as e:  # noqa: BLE001
                msg = str(e)
                err = msg[:500]
                if is_daily(msg) and re.search(r"429|rate|quota|limit|RESOURCE_EXHAUSTED", msg, re.I):
                    print(f"[{name}] STOP: provider daily limit: {msg[:200]}", flush=True)
                    return n_done
                if re.search(r"429|rate|quota|RESOURCE_EXHAUSTED|503|UNAVAILABLE|overloaded|timeout", msg, re.I):
                    time.sleep(min(90, retry_after(msg) or (5 * 2 ** attempt)) + 0.5)
                    continue
                break  # non-retryable (e.g. 400): record the error and move on
        latency = round(time.time() - started, 2)
        if err is not None and state.get("n_err", 0) < 5:
            state["n_err"] = state.get("n_err", 0) + 1
            print(f"[{name}] API ERROR on {row['call_id']}: {err[:300]}", flush=True)
        if err is not None:
            rec = dict(status="error", decision=None, approved=None, value=None, score=None, reason=None)
            text = None
        else:
            rec = parse_response(text, row["domain"], row.get("context", "general") or "general")
        if tt is None and err is None:
            tt = int(est)  # unknown usage: fall back to estimate so budgets stay conservative
        if tt:
            lim.record(tt, ct)
        out = dict(ts=time.time(), model_name=name, call_id=row["call_id"], status=rec["status"],
                   decision=rec["decision"], approved=rec["approved"], value=rec["value"],
                   score=rec["score"], reason=rec["reason"], raw_output=text, error=err, tokens=tt,
                   prompt_tokens=pt, completion_tokens=ct, latency_s=latency,
                   temperature=args.temperature, model_id=meta["id"], extras=state.get("extras"))
        with lock:
            out_fh.write(json.dumps(out) + "\n")
            out_fh.flush()
        n_done += 1
        if n_done % 25 == 0:
            rate = n_done / (time.time() - t0)
            print(f"[{name}] {n_done}/{len(todo)} this session ({rate*60:.1f}/min)", flush=True)
    print(f"[{name}] session finished: {n_done} calls", flush=True)
    return n_done


def summarize(args, rows, done, hist):
    rows = [r for r in rows if r["framing"] in args.framings]
    total = sum(1 for r in rows if int(r["pass_no"]) <= args.max_pass)
    print(f"\nManifest: {len(rows)} calls per model ({total} within --max-pass {args.max_pass}).")
    for name in args.models:
        d = sum(1 for r in rows if (name, r["call_id"]) in done and int(r["pass_no"]) <= args.max_pass)
        toks = [t for _, t in hist.get(name, [])]
        avg = sum(toks) / len(toks) if toks else None
        line = f"  {name:<24} done {d:>5}/{total}"
        if avg and MODELS[name]["provider"] == "groq" and args.daily_token_budget > 0:
            per_day = args.daily_token_budget / avg
            line += f" | avg {avg:.0f} tok/call -> ~{per_day:.0f} calls/day -> ~{(total - d) / per_day:.1f} days left"
        print(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="manifest_v3.csv")
    ap.add_argument("--out", default="results_v3.jsonl")
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--limit", type=int, default=0, help="max calls per model this session")
    ap.add_argument("--max-pass", type=int, default=99)
    ap.add_argument("--framings", default="implicit,explicit,anon,placebo",
                    help="comma list; use 'anon' for the gender-free calibration step")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-completion", type=int, default=1200)
    ap.add_argument("--gpt-oss-effort", default="low", choices=["low", "medium", "high", "default"])
    ap.add_argument("--qwen-effort", default="none", choices=["none", "default"],
                    help="'none' asks Qwen to skip its thinking phase (saves tokens); falls back if rejected")
    ap.add_argument("--retry-errors", action="store_true", help="retry calls that previously ended in an API error")
    ap.add_argument("--groq-rpm", type=int, default=25)
    ap.add_argument("--groq-tpm", type=int, default=7000)
    ap.add_argument("--daily-token-budget", type=int, default=170000)
    ap.add_argument("--daily-request-budget", type=int, default=900)
    ap.add_argument("--gemini-rpm", type=int, default=10)
    ap.add_argument("--gemini-rpd", type=int, default=900)
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    args.models = [m.strip() for m in args.models.split(",") if m.strip()]
    args.framings = [f.strip() for f in args.framings.split(",") if f.strip()]
    for m in args.models:
        if m not in MODELS:
            raise SystemExit(f"unknown model {m}; choose from {list(MODELS)}")

    with open(args.manifest, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    done, hist = load_history(args.out, args.retry_errors)
    summarize(args, rows, done, hist)
    if args.dry_run:
        return
    lock = threading.Lock()
    with open(args.out, "a", encoding="utf-8") as out_fh, ThreadPoolExecutor(len(args.models)) as ex:
        futs = [ex.submit(worker, m, rows, args, done, hist, lock, out_fh) for m in args.models]
        for f in futs:
            f.result()
    done, hist = load_history(args.out)
    summarize(args, rows, done, hist)


if __name__ == "__main__":
    main()
