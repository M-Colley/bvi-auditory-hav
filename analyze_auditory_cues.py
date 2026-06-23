#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_auditory_cues.py
========================
Reproducible acoustic characterisation of short auditory cues (e.g. the
non-speech notification sounds used in an automated-vehicle study).

For each input audio file the script measures, over the *active* (non-silent)
region:

  * active duration        - after -40 dB silence trimming
  * attack time            - 10-90 % rise time of the short-time RMS peak
  * time-to-peak           - time of the RMS maximum
  * decay (-20 dB)         - time from the RMS peak down to -20 dB
  * spectral centroid      - mean & median (perceived-brightness correlate)
  * spectral roll-off 95 % - mean
  * spectral bandwidth     - mean
  * spectral flatness      - mean (~0 tonal/harmonic, ~1 noise-like)
  * onset events           - count + times (librosa onset detection)
  * dominant partials      - top peaks of the time-averaged magnitude spectrum
  * fundamental estimates  - pYIN median + harmonic-product-spectrum (HPS)
  * per-onset pitch         - median YIN pitch per inter-onset segment (contour)

Outputs (written to --outdir, default ./cue_analysis_out):
  * cue_descriptors.csv      - one row per cue (the summary table)
  * cue_analysis.json        - full results incl. partials, contours,
                               parameters, and library versions (provenance)
  * <stem>_profile.{png,pdf} - per-cue envelope + log-frequency spectrogram
  * cue_profiles_grid.{png,pdf} - all cues stacked for comparison

METHODOLOGICAL NOTES (read before trusting the pitch fields)
  Automatic single-f0 trackers are unreliable for (a) broadband percussive
  onset transients and (b) polyphonic/struck sounds whose partials are not a
  single harmonic series. We therefore report the *partial structure* (which
  is model-free) alongside two f0 estimates, and recommend basing any pitch
  claim on the partials. pYIN can lock onto a sub-octave; HPS reduces but does
  not eliminate octave errors. The nearest-note labels are descriptive
  (closest equal-tempered note to a measured frequency), not assertions about
  intended musical notes.

  Descriptors characterise the *source files*. If stimuli were not
  loudness-matched and were played over participants' own hardware, absolute
  presentation level is not captured here.

REQUIREMENTS (tested versions in parentheses)
  python >= 3.9
  numpy        (>=1.24)
  scipy        (>=1.10)          # pulled in by librosa
  librosa      (==1.0; works on 0.10/0.11)
  soundfile / audioread + ffmpeg # for MP3 decoding (ffmpeg on PATH)
  matplotlib   (>=3.7)
Install:
  pip install "librosa>=0.10" matplotlib numpy
  # plus a system ffmpeg for MP3 (e.g. apt-get install ffmpeg)

USAGE
  python analyze_auditory_cues.py file1.mp3 file2.mp3 ...
  python analyze_auditory_cues.py *.mp3 --outdir results --no-pdf
  python analyze_auditory_cues.py sounds/ --recursive       # a directory

OPTIONAL LABELS
  To reproduce the study's event/level labels in the table, the filename stem
  is matched against DEFAULT_LABELS below; unmatched files fall back to their
  stem. Override or extend that dict as needed.

LICENSE
  Released under the MIT License. If you use this in academic work, please
  cite librosa (McFee et al., 2015) and pYIN (Mauch & Dixon, 2014).
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import warnings
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

import numpy as np

# ----------------------------------------------------------------------------
# Fixed analysis parameters (declared explicitly for reproducibility, rather
# than relying on library defaults that may change across versions).
# ----------------------------------------------------------------------------
SR = 44_100                 # resample everything to a common rate
TRIM_TOP_DB = 40            # silence-trim threshold for the "active" region
ENV_HOP = 256               # hop for the RMS amplitude envelope (timing)
SPEC_NFFT = 2048            # FFT size for centroid/rolloff/bandwidth/flatness
SPEC_HOP = 512              # hop for the above spectral features
PARTIAL_NFFT = 16_384       # high-resolution FFT for the averaged spectrum
PARTIAL_HOP = PARTIAL_NFFT // 4
PARTIAL_N = 8               # number of dominant partials to report
PARTIAL_MIN_SEP_HZ = 30.0   # merge spectral peaks closer than this
PARTIAL_FMIN_HZ = 60.0      # ignore sub-bass rumble below this
DECAY_DB = 20.0             # decay measured down to peak - DECAY_DB
ATTACK_LO, ATTACK_HI = 0.10, 0.90   # attack = 10%->90% of peak RMS
ONSET_DELTA = 0.06          # onset-detection peak-picking threshold
ONSET_WAIT_S = 0.05         # min spacing between detected onsets (seconds)
PYIN_FMIN, PYIN_FMAX = 80.0, 2000.0     # f0 search range (whole region)
CONTOUR_FMIN, CONTOUR_FMAX = 150.0, 4000.0  # per-onset pitch search range
HPS_HARMONICS = 5
SPEC_YLIM = (60, 16_000)    # spectrogram frequency axis (Hz)

# Study-specific event/level labels keyed by filename stem (optional).
DEFAULT_LABELS = {
    "insight-578":            ("start", "A"),
    "done-for-you-612":       ("stop/arrival", "A"),
    "i-demand-attention-244": ("pedestrian alert", "B"),
    "gesture-192":            ("manoeuvre", "B"),
}


# ----------------------------------------------------------------------------
# Result container
# ----------------------------------------------------------------------------
@dataclass
class CueResult:
    name: str
    file: str
    event: str = ""
    level: str = ""
    sample_rate: int = SR
    container_s: float = float("nan")
    active_s: float = float("nan")
    onset_active_s: float = float("nan")
    offset_active_s: float = float("nan")
    attack_s: float = float("nan")
    peak_s: float = float("nan")
    decay_db: float = DECAY_DB
    decay_s: float = float("nan")
    n_onsets: int = 0
    onset_times_s: list = field(default_factory=list)
    centroid_hz_mean: float = float("nan")
    centroid_hz_median: float = float("nan")
    rolloff95_hz_mean: float = float("nan")
    bandwidth_hz_mean: float = float("nan")
    flatness_mean: float = float("nan")
    pyin_f0_median_hz: float = float("nan")
    hps_f0_hz: float = float("nan")
    partials_hz_relmag: list = field(default_factory=list)   # [(hz, rel_mag), ...]
    per_onset_pitch: list = field(default_factory=list)      # [(t_s, hz_or_None, note_or_None), ...]


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def _safe_note(hz: Optional[float]) -> Optional[str]:
    import librosa
    if hz is None or not np.isfinite(hz) or hz <= 0:
        return None
    try:
        return str(librosa.hz_to_note(float(hz)))
    except Exception:
        return None


def envelope_timing(y, sr, idx):
    """Attack / peak / decay from the short-time RMS envelope.

    idx = (start_sample, end_sample) of the trimmed active region; timing is
    computed on the full signal but reported on the same clock.
    """
    import librosa
    rms = librosa.feature.rms(y=y, hop_length=ENV_HOP)[0]
    t = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=ENV_HOP)
    if rms.max() <= 0:
        return float("nan"), float("nan"), float("nan")
    pk = int(np.argmax(rms))
    peak_t = float(t[pk])
    lo = np.where(rms[:pk + 1] >= ATTACK_LO * rms.max())[0]
    hi = np.where(rms[:pk + 1] >= ATTACK_HI * rms.max())[0]
    attack = float(t[hi[0]] - t[lo[0]]) if (lo.size and hi.size) else float("nan")
    post = rms[pk:]
    thr = rms.max() * (10 ** (-DECAY_DB / 20.0))
    d = np.where(post <= thr)[0]
    decay = float(t[pk + d[0]] - peak_t) if d.size else float("nan")
    return attack, peak_t, decay


def spectral_descriptors(ya, sr):
    """Time-averaged spectral centroid / roll-off / bandwidth / flatness."""
    import librosa
    cent = librosa.feature.spectral_centroid(y=ya, sr=sr, n_fft=SPEC_NFFT, hop_length=SPEC_HOP)[0]
    roll = librosa.feature.spectral_rolloff(y=ya, sr=sr, n_fft=SPEC_NFFT, hop_length=SPEC_HOP, roll_percent=0.95)[0]
    bw = librosa.feature.spectral_bandwidth(y=ya, sr=sr, n_fft=SPEC_NFFT, hop_length=SPEC_HOP)[0]
    flat = librosa.feature.spectral_flatness(y=ya, n_fft=SPEC_NFFT, hop_length=SPEC_HOP)[0]
    return (float(cent.mean()), float(np.median(cent)),
            float(roll.mean()), float(bw.mean()), float(flat.mean()))


def dominant_partials(ya, sr):
    """Top peaks of the time-averaged magnitude spectrum (model-free pitch evidence)."""
    import librosa
    S = np.abs(librosa.stft(ya, n_fft=PARTIAL_NFFT, hop_length=PARTIAL_HOP)).mean(axis=1)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=PARTIAL_NFFT)
    if S.max() <= 0:
        return []
    m0 = S.max()
    out = []
    for p in np.argsort(-S):
        fr = float(freqs[p])
        if fr < PARTIAL_FMIN_HZ:
            continue
        if any(abs(fr - o[0]) < PARTIAL_MIN_SEP_HZ for o in out):
            continue
        out.append((fr, float(S[p] / m0)))
        if len(out) >= PARTIAL_N:
            break
    return sorted(out, key=lambda o: o[0])


def hps_f0(ya, sr):
    """Harmonic-product-spectrum fundamental estimate (octave-robust-ish)."""
    import librosa
    S = np.abs(librosa.stft(ya, n_fft=PARTIAL_NFFT, hop_length=PARTIAL_HOP)).mean(axis=1)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=PARTIAL_NFFT)
    hps = S.copy()
    for h in range(2, HPS_HARMONICS + 1):
        dec = S[::h]
        hps[:len(dec)] *= dec
    lo = int(np.searchsorted(freqs, PYIN_FMIN))
    hi = int(np.searchsorted(freqs, PYIN_FMAX))
    band = hps[lo:hi]
    if band.size == 0:
        return float("nan")
    return float(freqs[lo + int(np.argmax(band))])


def detect_onsets(y, sr):
    import librosa
    frames = librosa.onset.onset_detect(
        y=y, sr=sr, hop_length=ENV_HOP, backtrack=True,
        delta=ONSET_DELTA, wait=int(ONSET_WAIT_S * sr / ENV_HOP))
    times = librosa.frames_to_time(frames, sr=sr, hop_length=ENV_HOP)
    return frames, times


def per_onset_pitch(y, sr, onset_frames, end_sample):
    """Median YIN pitch within each inter-onset segment (the melodic contour)."""
    import librosa
    if onset_frames.size == 0:
        return []
    bounds = list((onset_frames * ENV_HOP)) + [end_sample]
    times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=ENV_HOP)
    out = []
    for i in range(len(onset_frames)):
        s, e = int(bounds[i]), int(bounds[i + 1])
        seg = y[s:e]
        pitch = None
        if len(seg) >= 1024:
            f0, _, _ = librosa.pyin(seg, fmin=CONTOUR_FMIN, fmax=CONTOUR_FMAX, sr=sr)
            v = f0[~np.isnan(f0)]
            if v.size:
                pitch = float(np.median(v))
        out.append((round(float(times[i]), 3), pitch, _safe_note(pitch)))
    return out


def analyze_file(path: Path) -> CueResult:
    import librosa
    y, sr = librosa.load(str(path), sr=SR, mono=True)
    stem = path.stem
    event, level = DEFAULT_LABELS.get(stem, ("", ""))
    res = CueResult(name=stem, file=str(path), event=event, level=level, sample_rate=sr)
    res.container_s = round(len(y) / sr, 3)

    yt, idx = librosa.effects.trim(y, top_db=TRIM_TOP_DB)
    res.onset_active_s = round(idx[0] / sr, 3)
    res.offset_active_s = round(idx[1] / sr, 3)
    res.active_s = round((idx[1] - idx[0]) / sr, 3)
    ya = y[idx[0]:idx[1]]

    attack, peak_t, decay = envelope_timing(y, sr, idx)
    res.attack_s, res.peak_s, res.decay_s = round(attack, 3), round(peak_t, 3), round(decay, 3)

    cmean, cmed, roll, bw, flat = spectral_descriptors(ya, sr)
    res.centroid_hz_mean = round(cmean, 1)
    res.centroid_hz_median = round(cmed, 1)
    res.rolloff95_hz_mean = round(roll, 1)
    res.bandwidth_hz_mean = round(bw, 1)
    res.flatness_mean = round(flat, 4)

    frames, times = detect_onsets(y, sr)
    res.n_onsets = int(len(frames))
    res.onset_times_s = [round(float(t), 3) for t in times]

    res.partials_hz_relmag = [(round(f, 1), round(m, 2)) for f, m in dominant_partials(ya, sr)]
    res.hps_f0_hz = round(hps_f0(ya, sr), 1)

    f0, _, _ = librosa.pyin(ya, fmin=PYIN_FMIN, fmax=PYIN_FMAX, sr=sr)
    f0v = f0[~np.isnan(f0)]
    res.pyin_f0_median_hz = round(float(np.median(f0v)), 1) if f0v.size else float("nan")

    res.per_onset_pitch = per_onset_pitch(y, sr, frames, idx[1])
    return res


# ----------------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------------
def _spectrogram_panel(ax, path, label, save_pdf=False):
    import librosa, librosa.display
    y, sr = librosa.load(str(path), sr=SR, mono=True)
    _, idx = librosa.effects.trim(y, top_db=TRIM_TOP_DB)
    ya = y[idx[0]:idx[1]]
    D = librosa.amplitude_to_db(np.abs(librosa.stft(ya, n_fft=4096, hop_length=ENV_HOP)), ref=np.max)
    img = librosa.display.specshow(D, sr=sr, hop_length=ENV_HOP, x_axis="time",
                                   y_axis="log", ax=ax, cmap="magma")
    ax.set_ylim(*SPEC_YLIM)
    ax.set_title(label, fontsize=10)
    rms = librosa.feature.rms(y=ya, hop_length=ENV_HOP)[0]
    te = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=ENV_HOP)
    ax2 = ax.twinx()
    if rms.max() > 0:
        ax2.plot(te, rms / rms.max(), color="#39d0ff", lw=1.1, alpha=0.9)
    ax2.set_ylim(0, 1.05)
    ax2.set_yticks([])
    ax2.set_ylabel("env", color="#39d0ff", fontsize=8)
    ax.set_xlim(0, max(1.2, float(te[-1]) if te.size else 1.2))
    return img


def plot_single(path: Path, res: CueResult, outdir: Path, save_pdf=True):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 3.2))
    _spectrogram_panel(ax, path, _panel_label(res))
    fig.tight_layout()
    png = outdir / f"{res.name}_profile.png"
    fig.savefig(png, dpi=140, bbox_inches="tight")
    if save_pdf:
        fig.savefig(outdir / f"{res.name}_profile.pdf", bbox_inches="tight")
    plt.close(fig)
    return png


def plot_grid(paths, results, outdir: Path, save_pdf=True):
    import matplotlib.pyplot as plt
    n = len(paths)
    fig, axes = plt.subplots(n, 1, figsize=(9, max(3.0, 2.75 * n)))
    if n == 1:
        axes = [axes]
    for ax, path, res in zip(axes, paths, results):
        _spectrogram_panel(ax, path, _panel_label(res))
    fig.suptitle("Acoustic profiles of the auditory cues "
                 "(log-frequency spectrogram + amplitude envelope)",
                 fontsize=11, y=0.997)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    png = outdir / "cue_profiles_grid.png"
    fig.savefig(png, dpi=140, bbox_inches="tight")
    if save_pdf:
        fig.savefig(outdir / "cue_profiles_grid.pdf", bbox_inches="tight")
    plt.close(fig)
    return png


def _panel_label(res: CueResult) -> str:
    if res.event:
        return f"{res.name} ({res.event}, level {res.level})" if res.level else f"{res.name} ({res.event})"
    return res.name


# ----------------------------------------------------------------------------
# Output writers
# ----------------------------------------------------------------------------
CSV_COLUMNS = [
    "name", "event", "level", "active_s", "attack_s", "peak_s", "decay_s",
    "centroid_hz_mean", "centroid_hz_median", "rolloff95_hz_mean",
    "bandwidth_hz_mean", "flatness_mean", "n_onsets",
    "pyin_f0_median_hz", "hps_f0_hz",
]


def write_csv(results, path: Path):
    import csv
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(CSV_COLUMNS)
        for r in results:
            d = asdict(r)
            w.writerow([d[c] for c in CSV_COLUMNS])


def write_json(results, params, path: Path):
    payload = {
        "parameters": params,
        "environment": _environment_info(),
        "results": [asdict(r) for r in results],
    }
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)


def _environment_info():
    info = {"python": sys.version.split()[0], "platform": platform.platform()}
    for mod in ("numpy", "scipy", "librosa", "matplotlib", "soundfile"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception:
            info[mod] = "not available"
    return info


def _params_dict():
    return {
        "sr": SR, "trim_top_db": TRIM_TOP_DB, "env_hop": ENV_HOP,
        "spec_nfft": SPEC_NFFT, "spec_hop": SPEC_HOP,
        "partial_nfft": PARTIAL_NFFT, "partial_hop": PARTIAL_HOP,
        "partial_n": PARTIAL_N, "partial_min_sep_hz": PARTIAL_MIN_SEP_HZ,
        "partial_fmin_hz": PARTIAL_FMIN_HZ, "decay_db": DECAY_DB,
        "attack_lo": ATTACK_LO, "attack_hi": ATTACK_HI,
        "onset_delta": ONSET_DELTA, "onset_wait_s": ONSET_WAIT_S,
        "pyin_fmin": PYIN_FMIN, "pyin_fmax": PYIN_FMAX,
        "contour_fmin": CONTOUR_FMIN, "contour_fmax": CONTOUR_FMAX,
        "hps_harmonics": HPS_HARMONICS,
    }


def print_summary(results):
    hdr = (f"{'cue':<24}{'dur':>7}{'atk':>7}{'dec':>7}"
           f"{'cent':>8}{'roll':>9}{'flat':>9}{'#on':>5}  pitch evidence")
    print("\n" + hdr)
    print("-" * len(hdr))
    for r in results:
        parts = ", ".join(f"{f:.0f}" for f, _ in r.partials_hz_relmag[:4])
        print(f"{_panel_label(r)[:23]:<24}"
              f"{r.active_s:>7.2f}{r.attack_s:>7.2f}{r.decay_s:>7.2f}"
              f"{r.centroid_hz_mean:>8.0f}{r.rolloff95_hz_mean:>9.0f}"
              f"{r.flatness_mean:>9.3f}{r.n_onsets:>5d}  [{parts} Hz]")
    print("\nPer-onset pitch contours:")
    for r in results:
        seq = " ".join(
            (note if note else "--") for _, _, note in r.per_onset_pitch)
        print(f"  {r.name:<24} {seq}")


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def collect_files(inputs, recursive):
    exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".aiff", ".aif"}
    files = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            it = p.rglob("*") if recursive else p.glob("*")
            files += sorted(q for q in it if q.suffix.lower() in exts)
        elif p.is_file():
            files.append(p)
        else:
            print(f"warning: not found, skipping: {p}", file=sys.stderr)
    # de-duplicate, preserve order
    seen, uniq = set(), []
    for f in files:
        if f not in seen:
            seen.add(f)
            uniq.append(f)
    return uniq


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Reproducible acoustic characterisation of short auditory cues.")
    ap.add_argument("inputs", nargs="+", help="audio files and/or directories")
    ap.add_argument("--outdir", default="cue_analysis_out", help="output directory")
    ap.add_argument("--recursive", action="store_true",
                    help="recurse into directories")
    ap.add_argument("--no-plots", action="store_true", help="skip figures")
    ap.add_argument("--no-pdf", action="store_true",
                    help="save PNG figures only (no PDF)")
    ap.add_argument("--no-single", action="store_true",
                    help="skip per-cue figures, only the grid")
    args = ap.parse_args(argv)

    warnings.filterwarnings("ignore")  # silence librosa/pyin runtime warnings
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    files = collect_files(args.inputs, args.recursive)
    if not files:
        print("error: no audio files to analyse.", file=sys.stderr)
        return 2

    results = []
    for f in files:
        print(f"analysing: {f.name}")
        results.append(analyze_file(f))

    write_csv(results, outdir / "cue_descriptors.csv")
    write_json(results, _params_dict(), outdir / "cue_analysis.json")

    if not args.no_plots:
        import matplotlib
        matplotlib.use("Agg")
        if not args.no_single:
            for f, r in zip(files, results):
                plot_single(f, r, outdir, save_pdf=not args.no_pdf)
        if len(files) > 1:
            plot_grid(files, results, outdir, save_pdf=not args.no_pdf)

    print_summary(results)
    print(f"\nWrote: {outdir/'cue_descriptors.csv'}, {outdir/'cue_analysis.json'}"
          + ("" if args.no_plots else f", and figures in {outdir}/"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
