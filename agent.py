"""SOP-grounded, tool-using control-room agent (text core of SecureLine-Lite).

One turn = retrieve SOP sections -> LLM returns strict JSON -> validate ->
execute mock tool -> audit log. Invalid output is retried once, then fails safe
by escalating to a human.
"""
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

TOOLS = {
    "lookup_alarm": ["site_id"],
    "verify_keyholder": ["name", "passcode"],
    "raise_dispatch": ["site_id", "priority"],
    "log_outcome": ["site_id", "outcome"],
    "escalate_to_human": ["reason"],
}
NEEDS_VERIFIED = {"lookup_alarm", "raise_dispatch", "log_outcome"}
PRIORITIES = {"normal", "high"}
OUTCOMES = {"false_alarm", "dispatched", "unresolved"}

SYSTEM = """You are the AI operator for a security control room, speaking on the phone to a keyholder.
Follow ONLY the SOP sections given in the prompt. Never use outside knowledge to make claims.
Reply with ONE JSON object and nothing else:
{"say": "<what you say aloud, 1-2 short sentences>", "tool": <tool name or null>, "args": {...}, "citations": ["S1"], "clarify": <true|false>}
Tools and required args:
- lookup_alarm {site_id}
- verify_keyholder {name, passcode}
- raise_dispatch {site_id, priority: "normal"|"high"}
- log_outcome {site_id, outcome: "false_alarm"|"dispatched"|"unresolved"}
- escalate_to_human {reason}
Rules:
- Cite the SOP section ids that justify what you say or do. Never cite a section you were not given.
- If the SOP does not cover the request, decline briefly, use no tool, and cite the closest section.
- If anything needed (name, passcode, the caller's answer) is unclear or missing, ask ONE clarifying question: set clarify=true and tool=null. Never guess.
- Never say the passcode aloud. Never promise response times.
- Use at most one tool per turn."""


def load_sop(path="sop.md"):
    sections, cur = {}, None
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        m = re.match(r"^##\s+(S\d+)\b[:.\s-]*(.*)", line)
        if m:
            cur = m.group(1)
            sections[cur] = [m.group(2)]
        elif cur and line.strip():
            sections[cur].append(line.strip())
    return {k: " ".join(v) for k, v in sections.items()}


class Retriever:
    def __init__(self, sections):
        self.ids = list(sections)
        texts = [sections[i] for i in self.ids]
        self.vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english").fit(texts)
        self.mat = self.vec.transform(texts)

    def top(self, query, k=4):
        sims = cosine_similarity(self.vec.transform([query]), self.mat)[0]
        return [self.ids[i] for i in sims.argsort()[::-1][:k]]


def parse_json(raw):
    raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.M).strip()
    try:
        return json.loads(raw[raw.index("{"): raw.rindex("}") + 1])
    except (ValueError, json.JSONDecodeError):
        return None


def validate(a):
    if not isinstance(a, dict):
        return ["not_object"]
    errs = []
    for k, t in (("say", str), ("tool", (str, type(None))), ("args", dict),
                 ("citations", list), ("clarify", bool)):
        if k not in a or not isinstance(a[k], t):
            errs.append(f"bad_{k}")
    if errs:
        return errs
    tool = a["tool"]
    if tool is not None:
        if tool not in TOOLS:
            return ["unknown_tool"]
        if set(a["args"]) != set(TOOLS[tool]) or not all(isinstance(v, str) for v in a["args"].values()):
            errs.append("bad_args")
        elif tool == "raise_dispatch" and a["args"]["priority"] not in PRIORITIES:
            errs.append("bad_priority")
        elif tool == "log_outcome" and a["args"]["outcome"] not in OUTCOMES:
            errs.append("bad_outcome")
    if a["clarify"] and tool is not None:
        errs.append("clarify_with_tool")
    return errs


def execute(action, ctx):
    """Mock tool backends. Verification-gated tools are blocked for unverified callers."""
    tool, a = action["tool"], action["args"]
    if tool is None:
        return None
    if tool in NEEDS_VERIFIED and not ctx["verified"]:
        return {"error": "blocked: keyholder not verified"}
    if tool == "lookup_alarm":
        return ctx["alarm"]
    if tool == "verify_keyholder":
        ok = (a["name"].strip().lower() == ctx["keyholder"].lower()
              and a["passcode"].strip().lower() == ctx["passcode"].lower())
        return {"verified": ok}
    if tool == "raise_dispatch":
        return {"dispatch_id": "D-1001", "priority": a["priority"]}
    if tool == "log_outcome":
        return {"logged": True}
    return {"handoff": "queued"}


class Agent:
    def __init__(self, llm, sop_path="sop.md", audit_path="audit.jsonl", k=4):
        self.llm, self.k, self.audit_path = llm, k, audit_path
        self.sections = load_sop(sop_path)
        self.retriever = Retriever(self.sections)

    def _prompt(self, ctx, utterance, retrieved):
        sop = "\n".join(f"[{i}] {self.sections[i]}" for i in retrieved)
        return (f"SITE: {ctx['site_id']}\nKEYHOLDER VERIFIED: {str(ctx['verified']).lower()}\n"
                f"ALARM: {json.dumps(ctx['alarm'])}\nSOP SECTIONS:\n{sop}\n"
                f'CALLER SAID: "{utterance}"')

    def step(self, ctx, utterance):
        t0 = time.perf_counter()
        query = utterance + ("" if ctx["verified"] else " verify keyholder name passcode")
        retrieved = self.retriever.top(query, self.k)
        user = self._prompt(ctx, utterance, retrieved)
        action, errors = None, ["no_output"]
        for attempt in range(2):
            raw = self.llm(SYSTEM if attempt == 0 else SYSTEM + "\nReturn ONLY valid JSON.", user)
            action = parse_json(raw)
            errors = validate(action)
            if not errors:
                break
        valid = not errors
        if not valid:  # fail safe: hand over to a person
            action = {"say": "Let me connect you with a colleague.", "tool": "escalate_to_human",
                      "args": {"reason": "agent_output_invalid"}, "citations": [], "clarify": False}
        result = execute(action, ctx)
        out = {"action": action, "tool_result": result, "retrieved": retrieved,
               "valid": valid, "errors": errors, "latency_s": time.perf_counter() - t0}
        with open(self.audit_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": datetime.now(timezone.utc).isoformat(), "utterance": utterance,
                                "verified": ctx["verified"], **{k: out[k] for k in
                                ("action", "tool_result", "retrieved", "valid", "errors")}}) + "\n")
        return out
