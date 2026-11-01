"""Entity-focused ASR evaluation under telephone-band degradation.

Pipeline: synthesize test sentences with local TTS voices -> degrade to phone quality
(8 kHz, 300-3400 Hz band, optional noise, G.711 mu-law) -> faster-whisper -> WER and
entity error rate (EER) with bootstrap CIs and paired clean-vs-condition differences.
"""
import argparse
import csv
import re
from math import gcd
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, resample_poly, sosfilt

SENTENCES = [
    ("The alarm was triggered at the Sabarmati Riverfront warehouse in zone three.", ["sabarmati riverfront"]),
    ("Please send a patrol to Vastrapur Lake and check the east gate.", ["vastrapur"]),
    ("The keyholder for the Bopal depot is not answering the phone.", ["bopal", "keyholder"]),
    ("We detected a perimeter breach near the Thaltej substation.", ["perimeter breach", "thaltej"]),
    ("The CCTV camera on Satellite Road shows no movement.", ["cctv", "satellite road"]),
    ("Dispatch an officer to the Gandhinagar data centre immediately.", ["gandhinagar"]),
    ("The passcode check failed twice for the Navrangpura office.", ["navrangpura"]),
    ("Door contact sensor four was opened at the Maninagar yard.", ["maninagar"]),
    ("Please confirm the keyholder is on site at Ashram Road.", ["keyholder", "ashram road"]),
    ("The motion detector in the Naroda storage unit has reset.", ["naroda"]),
    ("A false alarm was logged for the Prahladnagar tower.", ["prahladnagar"]),
    ("Escalate to the duty supervisor about the Chandkheda gate.", ["chandkheda"]),
    ("The tamper alert at Sarkhej came from the control panel.", ["sarkhej", "tamper alert"]),
    ("Patrol unit two is heading to the Ellisbridge car park.", ["ellisbridge"]),
    ("The glass break sensor at Bodakdev activated at two fourteen.", ["bodakdev", "glass break"]),
    ("Confirm whether the intruder alarm at Gota is genuine.", ["gota", "intruder alarm"]),
    ("The roller shutter at the Vatva industrial estate is open.", ["vatva"]),
    ("Please give your site passcode before I share any details.", ["passcode"]),
    ("An armed response is not available, so I am escalating to a person.", ["armed response"]),
    ("The fire panel at the Odhav warehouse reports a fault.", ["odhav", "fire panel"]),
]
CONDITIONS = {"clean": None, "phone": "clean_phone", "phone_snr20": 20, "phone_snr10": 10, "phone_snr5": 5}


# ---------- audio ----------
def load_wav16(path):
    sr, x = wavfile.read(path)
    x = x.astype(np.float32)
    if x.dtype != np.float32 or np.abs(x).max() > 1.5:
        x = x / 32768.0
    if x.ndim > 1:
        x = x.mean(1)
    g = gcd(sr, 16000)
    return resample_poly(x, 16000 // g, sr // g).astype(np.float32)


def mulaw(x, mu=255):
    x = np.clip(x, -1, 1)
    y = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
    y = np.round((y + 1) / 2 * mu) / mu * 2 - 1  # 8-bit quantisation
    return np.sign(y) * ((1 + mu) ** np.abs(y) - 1) / mu


def phone(x16, snr_db=None, seed=0):
    """16 kHz float audio -> narrowband telephone audio, returned at 16 kHz for the ASR."""
    x8 = resample_poly(x16, 1, 2)
    x8 = sosfilt(butter(4, [300, 3400], btype="band", fs=8000, output="sos"), x8)
    if snr_db is not None:
        rng = np.random.default_rng(seed)
        p = np.mean(x8 ** 2) + 1e-12
        x8 = x8 + rng.standard_normal(len(x8)) * np.sqrt(p / 10 ** (snr_db / 10))
    x8 = x8 / (np.max(np.abs(x8)) + 1e-9) * 0.9
    return resample_poly(mulaw(x8), 2, 1).astype(np.float32)


def apply_condition(x16, cond):
    spec = CONDITIONS[cond]
    if spec is None:
        return x16
    return phone(x16, None if spec == "clean_phone" else spec)


# ---------- metrics ----------
def norm(s):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", s.lower())).strip()


def wer_counts(ref, hyp):
    r, h = norm(ref).split(), norm(hyp).split()
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev, d[j] = d[j], cur
    return d[len(h)], len(r)


def entity_counts(entities, hyp):
    hn = norm(hyp)
    return sum(norm(e) not in hn for e in entities), len(entities)


def ratio_ci(num, den, n=2000, seed=0):
    num, den = np.asarray(num, float), np.asarray(den, float)
    idx = np.random.default_rng(seed).integers(0, len(num), (n, len(num)))
    r = num[idx].sum(1) / np.maximum(den[idx].sum(1), 1)
    return num.sum() / max(den.sum(), 1), *np.percentile(r, [2.5, 97.5])


def paired_diff_ci(na, da, nb, db, n=2000, seed=0):
    """CI for ratio(b) - ratio(a) on the same items (paired bootstrap)."""
    na, da, nb, db = (np.asarray(v, float) for v in (na, da, nb, db))
    idx = np.random.default_rng(seed).integers(0, len(na), (n, len(na)))
    diff = nb[idx].sum(1) / np.maximum(db[idx].sum(1), 1) - na[idx].sum(1) / np.maximum(da[idx].sum(1), 1)
    return float(nb.sum() / db.sum() - na.sum() / da.sum()), *np.percentile(diff, [2.5, 97.5])


# ---------- TTS + ASR ----------
def synthesize(outdir, n_voices=2, limit=None):
    import pyttsx3
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    engine = pyttsx3.init()
    voices = engine.getProperty("voices")[:n_voices]
    items = SENTENCES[:limit]
    for vi, v in enumerate(voices):
        engine.setProperty("voice", v.id)
        for si, (text, _) in enumerate(items):
            p = outdir / f"v{vi}_s{si:02d}.wav"
            if not p.exists():
                engine.save_to_file(text, str(p))
                engine.runAndWait()
    return [f"v{i}:{v.name}" for i, v in enumerate(voices)]


def load_asr(size, device):
    from faster_whisper import WhisperModel
    if device == "auto":
        import ctranslate2
        device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
    return WhisperModel(size, device=device, compute_type="float16" if device == "cuda" else "int8")


def transcribe(model, audio):
    segs, _ = model.transcribe(audio, language="en", beam_size=5)
    return " ".join(s.text for s in segs).strip()


def run(model, audio_dir, voice_names, limit=None, transcribe_fn=transcribe):
    rows = []
    items = SENTENCES[:limit]
    for vi, vname in enumerate(voice_names):
        for si, (text, ents) in enumerate(items):
            x = load_wav16(Path(audio_dir) / f"v{vi}_s{si:02d}.wav")
            for cond in CONDITIONS:
                hyp = transcribe_fn(model, apply_condition(x, cond))
                we, wn = wer_counts(text, hyp)
                em, et = entity_counts(ents, hyp)
                rows.append(dict(voice=vname, cond=cond, sid=si, ref=text, hyp=hyp,
                                 w_err=we, w_n=wn, e_miss=em, e_n=et))
    return rows


def summarize(rows):
    conds = list(CONDITIONS)
    def sel(c, v=None):
        return [r for r in rows if r["cond"] == c and (v is None or r["voice"] == v)]
    def col(rs, k):
        return [r[k] for r in rs]
    lines = ["| Condition | WER [95% CI] | Entity error rate [95% CI] | EER change vs clean (paired, 95% CI) |",
             "|---|---|---|---|"]
    base = sel("clean")
    for c in conds:
        rs = sel(c)
        w = ratio_ci(col(rs, "w_err"), col(rs, "w_n"))
        e = ratio_ci(col(rs, "e_miss"), col(rs, "e_n"))
        d = "-" if c == "clean" else "{:+.1%} [{:+.1%}, {:+.1%}]".format(
            *paired_diff_ci(col(base, "e_miss"), col(base, "e_n"), col(rs, "e_miss"), col(rs, "e_n")))
        lines.append(f"| {c} | {w[0]:.1%} [{w[1]:.1%}, {w[2]:.1%}] | {e[0]:.1%} [{e[1]:.1%}, {e[2]:.1%}] | {d} |")
    voices = sorted({r["voice"] for r in rows})
    lines += ["", "Entity error rate by voice:", "", "| Voice | " + " | ".join(conds) + " |",
              "|---|" + "---|" * len(conds)]
    for v in voices:
        cells = []
        for c in conds:
            rs = sel(c, v)
            cells.append("{:.1%}".format(sum(col(rs, "e_miss")) / max(sum(col(rs, "e_n")), 1)))
        lines.append(f"| {v} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="small")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--audio-dir", default="data/audio")
    ap.add_argument("--voices", type=int, default=2)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    names = synthesize(a.audio_dir, a.voices, a.limit)
    rows = run(load_asr(a.model, a.device), a.audio_dir, names, a.limit)
    with open("results_asr.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    md = f"**ASR:** faster-whisper `{a.model}` | **sentences:** {len(SENTENCES[:a.limit])} | **voices:** {len(names)}\n\n" + summarize(rows)
    Path("results_asr.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
