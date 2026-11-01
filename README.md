# SecureLine-Lite

A small, reproducible voice-agent project modelled on a security control-room workflow: a keyholder
phones in about an alarm, the system transcribes the call, and an agent answers **only from a written
SOP**, calls tools with validated arguments, escalates to a human when required, and logs every step.
The project is built around **evaluation first**: how correct, how grounded and how robust is it?

```
caller audio -> (phone-band degradation) -> faster-whisper ASR -> SOP retrieval (TF-IDF)
             -> LLM (strict JSON) -> validation -> tool execution (mock) -> audit log -> reply (+TTS)
```

## What it measures

**Voice front end (`asr_eval.py`)**
- Word error rate **and entity error rate** (place names, security terms such as "keyholder", "perimeter breach")
- Five conditions: clean, 8 kHz phone band with G.711 mu-law, and phone + noise at 20 / 10 / 5 dB SNR
- Bootstrap 95% CIs, plus a **paired** bootstrap of the entity-error change versus clean
- Breakdown by TTS voice

**Agent (`eval_agent.py`, 15 scripted scenarios in `scenarios.json`)**
- Tool-selection accuracy, argument accuracy, schema validity
- Missed escalation and false escalation rates
- Clarifying-question accuracy (ask instead of guess)
- **Unsupported-claim rate**: reply has no citation, or cites an SOP section it was not given
- Unsafe tool calls (acting for an unverified caller), retrieval hit rate, latency p50 / p95 / p99
- Bootstrap CIs over scenarios, outcomes averaged over repeated runs

**Safety design:** verification-gated tools, schema-checked arguments, one retry then fail-safe
escalation to a human, JSONL audit log (`audit.jsonl`), passcode never placed in the prompt.

## Run it

```bash
pip install -r requirements.txt
pytest -q                                   # 8 tests, no models needed

python eval_agent.py --llm mock             # offline smoke test of the harness (a keyword stub, not a model)
export GEMINI_API_KEY=...                   # Windows PowerShell: $env:GEMINI_API_KEY="..."
python eval_agent.py --llm gemini --runs 5  # real agent results -> results_agent.md

python asr_eval.py --model small            # synthesizes audio, degrades it, runs Whisper -> results_asr.md/.csv
python call_demo.py data/audio/v0_s00.wav --phone --verified --speak
```
`--model base` is faster; `small` fits comfortably in 4 GB VRAM. Set `GEMINI_MODEL` if you want a different Gemini model.
On Windows, `pyttsx3` uses the built-in SAPI5 voices, so no extra TTS install is needed.

## Results

Run the commands above and paste the generated tables here. Do not report numbers you have not produced.

| Experiment | Result file |
|---|---|
| Agent evaluation | `results_agent.md` |
| ASR entity error vs. phone-band noise | `results_asr.md`, `results_asr.csv` |

## Limitations (state these in interviews)

- Test audio is **synthetic** (local TTS, two voices). It says nothing about real accents or real callers.
  The next step is real recordings with a transcription protocol and inter-annotator agreement.
- 15 scenarios and 20 sentences give wide confidence intervals. Grow both before drawing conclusions.
- Agent scenarios are single-turn. Multi-turn dialogue, barge-in and streaming are not implemented.
- Entity matching is exact after normalisation, so "key holder" counts as a miss for "keyholder".
  That is deliberate (it is a real domain failure), but a fuzzy variant would be a useful ablation.
- Tools are mocks. The SOP is fictional.

## Next steps

LoRA fine-tune Whisper on domain vocabulary and re-run `asr_eval.py` on held-out sentences and voices;
streaming TTS with sentence-level chunking and latency percentiles under concurrent calls;
constrained decoding for the JSON output; a short NIST AI RMF mapping for acceptance criteria and monitoring.
