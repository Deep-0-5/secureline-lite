"""LLM backends. GeminiLLM is the real one; MockLLM is a keyword stub so the
pipeline and the evaluation harness can run offline (it is NOT a model)."""
import json
import os
import re
import time


class GeminiLLM:
    def __init__(self, model=None, pause=0):
        from google import genai
        self.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        self.model = model or os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
        self.pause = pause          # seconds to wait between requests

    def __call__(self, system, user):
        max_retries = 5
        for attempt in range(1, max_retries + 1):
            try:
                if self.pause and attempt == 1:
                    time.sleep(self.pause)
                r = self.client.models.generate_content(
                    model=self.model,
                    contents=user,
                    config={"system_instruction": system, "temperature": 0.2,
                            "response_mime_type": "application/json"},
                )
                return r.text
            except Exception as e:
                err = str(e)
                if ("429" in err or "503" in err) and attempt < max_retries:
                    wait = 40 * attempt
                    print(f"[retry] {err[:80]}... retrying in {wait}s (attempt {attempt}/{max_retries})")
                    time.sleep(wait)
                else:
                    raise


EMERG = re.compile(r"someone (is )?in|intruder|break-?in|glass|door (is )?open|ambulance|dizzy|fire|hurt|injur|weapon", re.I)
HUMAN = re.compile(r"real person|human|supervisor|manager", re.I)
DECLINE = re.compile(r"guarantee|promise|neighbo|other (client|customer)|police", re.I)
DETAILS = re.compile(r"what (caused|happened)|which zone|what time|alarm details", re.I)
UNSURE = re.compile(r"not sure|maybe|probably|i think|dunno", re.I)
FALSE = re.compile(r"accident|false alarm|everything (looks|is) fine|set it off", re.I)
AWAY = re.compile(r"out of town|hours away|can'?t get|cannot get|send someone|not on site", re.I)
NAME = re.compile(r"(?:this is|my name is|i am|i'm)\s+([a-z]+ [a-z]+)", re.I)
PASS = re.compile(r"(?:passcode|pass code|password|word) is\s+(\w+)", re.I)


class MockLLM:
    def __call__(self, system, user):
        verified = "KEYHOLDER VERIFIED: true" in user
        site = re.search(r"SITE: (\S+)", user).group(1)
        said = re.search(r'CALLER SAID: "(.*)"', user, re.S).group(1)
        seen = set(re.findall(r"^\[(S\d+)\]", user, re.M))

        def out(say, tool=None, args=None, cites=(), clarify=False):
            return json.dumps({"say": say, "tool": tool, "args": args or {},
                               "citations": [c for c in cites if c in seen],
                               "clarify": clarify})

        if EMERG.search(said) or HUMAN.search(said):
            return out("I'm connecting you to a person now.", "escalate_to_human",
                       {"reason": said[:80]}, ["S4"])
        if not verified:
            n, p = NAME.search(said), PASS.search(said)
            if n and p:
                return out("Thank you, checking that now.", "verify_keyholder",
                           {"name": n.group(1).title(), "passcode": p.group(1)}, ["S1"])
            return out("Before I can help, please tell me your full name and the site passcode.",
                       cites=["S1", "S6"], clarify=True)
        if DECLINE.search(said):
            return out("I'm sorry, I can't help with that.", cites=["S7"])
        if DETAILS.search(said):
            return out("Let me pull up the alarm details.", "lookup_alarm", {"site_id": site}, ["S2"])
        if UNSURE.search(said):
            return out("Can you tell me clearly whether you are on site right now?", cites=["S6"], clarify=True)
        if FALSE.search(said):
            return out("Understood, I'll log this as a false alarm.", "log_outcome",
                       {"site_id": site, "outcome": "false_alarm"}, ["S5"])
        if AWAY.search(said):
            return out("I'll arrange a patrol to check the site.", "raise_dispatch",
                       {"site_id": site, "priority": "normal"}, ["S3"])
        return out("Could you tell me a bit more about the situation?", cites=["S6"], clarify=True)
