import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import agent
import asr_eval
import eval_agent
from llm import MockLLM


def test_sop_sections():
    assert list(agent.load_sop()) == [f"S{i}" for i in range(1, 8)]


def test_validate_rejects_bad_output():
    good = {"say": "ok", "tool": "lookup_alarm", "args": {"site_id": "S-1"}, "citations": ["S2"], "clarify": False}
    assert agent.validate(good) == []
    assert "bad_args" in agent.validate({**good, "args": {}})
    assert "unknown_tool" in agent.validate({**good, "tool": "delete_everything"})
    assert "clarify_with_tool" in agent.validate({**good, "clarify": True})
    assert agent.validate("not json") == ["not_object"]


def test_unverified_caller_is_blocked():
    ctx = {**eval_agent.CTX, "verified": False}
    out = agent.execute({"tool": "raise_dispatch", "args": {"site_id": "S-014", "priority": "normal"}}, ctx)
    assert "blocked" in out["error"]


def test_invalid_llm_output_fails_safe(tmp_path):
    a = agent.Agent(lambda s, u: "garbage", audit_path=str(tmp_path / "a.jsonl"))
    res = a.step({**eval_agent.CTX, "verified": True}, "hello")
    assert res["action"]["tool"] == "escalate_to_human" and not res["valid"]


def test_wer_and_entities():
    assert asr_eval.wer_counts("a b c d", "a x c") == (2, 4)
    assert asr_eval.entity_counts(["key holder", "gota"], "the key holder at Gota") == (0, 2)
    assert asr_eval.entity_counts(["keyholder"], "the key holder") == (1, 1)


def test_phone_degradation():
    x = np.random.default_rng(0).standard_normal(16000).astype(np.float32) * 0.1
    y = asr_eval.phone(x, snr_db=10)
    assert y.shape == x.shape and np.isfinite(y).all() and np.abs(y).max() <= 1.0


def test_bootstrap_ci_contains_estimate():
    m, lo, hi = asr_eval.ratio_ci([1, 0, 2, 1, 0, 3], [4, 4, 4, 4, 4, 4])
    assert lo <= m <= hi


def test_mock_agent_eval(tmp_path):
    sc = json.loads(Path("scenarios.json").read_text(encoding="utf-8"))
    s = eval_agent.evaluate(MockLLM(), sc, runs=1, audit_path=str(tmp_path / "a.jsonl"))
    assert s["tool_acc"]["mean"] >= 0.9 and s["unsafe_tool_call"]["mean"] == 0
