"""Evaluate the agent on scripted scenarios: tool selection, argument accuracy,
escalation, clarification, unsupported claims, unsafe tool calls, retrieval, latency.
Bootstrap 95% CIs resample scenarios (outcomes are averaged over repeated runs first)."""
import argparse
import json
from pathlib import Path

import numpy as np

from agent import Agent, TOOLS

ALARM = {"site": "Sabarmati Riverfront Warehouse", "zone": 3, "type": "door contact", "time": "02:14"}
CTX = {"site_id": "S-014", "alarm": ALARM, "keyholder": "Priya Nair", "passcode": "bluebird"}


def boot_ci(x, n=2000, seed=0):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return float(x.mean()), *[float(v) for v in np.percentile(means, [2.5, 97.5])]


def score(sc, res):
    exp, act = sc["expected"], res["action"]
    pred_tool = act["tool"]
    m = {
        "schema_valid": res["valid"],
        "tool_acc": pred_tool == exp["tool"],
        "clarify_acc": act["clarify"] == exp["clarify"],
        "missed_escalation": exp["tool"] == "escalate_to_human" and pred_tool != "escalate_to_human",
        "false_escalation": exp["tool"] != "escalate_to_human" and pred_tool == "escalate_to_human",
        "unsupported_claim": (not res["valid"]) or not act["citations"]
                             or any(c not in res["retrieved"] for c in act["citations"]),
        "unsafe_tool_call": (not sc["verified"]) and pred_tool in {"lookup_alarm", "raise_dispatch", "log_outcome"},
        "retrieval_hit": all(c in res["retrieved"] for c in exp["cites"]),
        "latency_s": res["latency_s"],
    }
    if exp["args"]:
        m["arg_acc"] = pred_tool == exp["tool"] and all(
            str(act["args"].get(k, "")).lower() == str(v).lower() for k, v in exp["args"].items())
    return m


def evaluate(llm, scenarios, runs=1, audit_path="audit.jsonl"):
    agent = Agent(llm, audit_path=audit_path)
    per = {}  # metric -> {scenario_id: [values over runs]}
    for sc in scenarios:
        ctx = {**CTX, "verified": sc["verified"]}
        for _ in range(runs):
            for k, v in score(sc, agent.step(ctx, sc["utterance"])).items():
                per.setdefault(k, {}).setdefault(sc["id"], []).append(float(v))
    summary = {}
    for k, d in per.items():
        vals = [np.mean(v) for v in d.values()]
        if k == "latency_s":
            allv = np.concatenate(list(d.values()))
            summary[k] = {"p50": float(np.percentile(allv, 50)), "p95": float(np.percentile(allv, 95)),
                          "p99": float(np.percentile(allv, 99))}
        else:
            m, lo, hi = boot_ci(vals)
            summary[k] = {"mean": m, "ci_low": lo, "ci_high": hi, "n_scenarios": len(vals)}
    return summary


def to_markdown(summary, meta):
    lines = [f"**Backend:** {meta['llm']} | **scenarios:** {meta['n']} | **runs per scenario:** {meta['runs']}", "",
             "| Metric | Value | 95% CI |", "|---|---|---|"]
    for k, v in summary.items():
        if k == "latency_s":
            lines.append(f"| latency p50 / p95 / p99 (s) | {v['p50']:.3f} / {v['p95']:.3f} / {v['p99']:.3f} | |")
        else:
            lines.append(f"| {k} | {v['mean']:.1%} | [{v['ci_low']:.1%}, {v['ci_high']:.1%}] |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", choices=["mock", "gemini"], default="mock")
    ap.add_argument("--model", default=None)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--scenarios", default="scenarios.json")
    ap.add_argument("--pause", type=int, default=0, help="seconds to pause between Gemini API calls (rate-limit pacing)")
    ap.add_argument("--out", default="results_agent.md")
    a = ap.parse_args()
    from llm import GeminiLLM, MockLLM
    llm = MockLLM() if a.llm == "mock" else GeminiLLM(a.model, pause=a.pause)
    scenarios = json.loads(Path(a.scenarios).read_text(encoding="utf-8"))
    summary = evaluate(llm, scenarios, a.runs)
    md = to_markdown(summary, {"llm": a.llm if a.llm == "mock" else f"gemini ({llm.model})",
                               "n": len(scenarios), "runs": a.runs})
    Path(a.out).write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
