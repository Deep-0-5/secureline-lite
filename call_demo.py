"""End-to-end demo: wav -> (optional phone degradation) -> ASR -> SOP-grounded agent -> reply (+ optional TTS)."""
import argparse

from agent import Agent
from asr_eval import load_asr, load_wav16, phone, transcribe
from eval_agent import CTX
from llm import GeminiLLM, MockLLM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("--verified", action="store_true", help="treat the keyholder as already verified")
    ap.add_argument("--phone", action="store_true", help="degrade audio to telephone quality first")
    ap.add_argument("--llm", choices=["mock", "gemini"], default="gemini")
    ap.add_argument("--asr-model", default="small")
    ap.add_argument("--speak", action="store_true")
    a = ap.parse_args()
    x = load_wav16(a.wav)
    if a.phone:
        x = phone(x, snr_db=15)
    text = transcribe(load_asr(a.asr_model, "auto"), x)
    print(f"[ASR] {text}")
    agent = Agent(MockLLM() if a.llm == "mock" else GeminiLLM())
    res = agent.step({**CTX, "verified": a.verified}, text)
    act = res["action"]
    print(f"[AGENT] {act['say']}\n[TOOL] {act['tool']} {act['args']} -> {res['tool_result']}\n[CITES] {act['citations']}")
    if a.speak:
        import pyttsx3
        e = pyttsx3.init()
        e.say(act["say"])
        e.runAndWait()


if __name__ == "__main__":
    main()
