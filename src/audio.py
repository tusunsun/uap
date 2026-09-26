"""Procedural trailer score + sound design, locked to the picture edit in timeline.py.

    python3 src/audio.py            -> build/score.wav
"""
import math
import os
import sys

import numpy as np
from scipy.io import wavfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dsp import (SR, bp, brown, convolve_st, env_adsr, env_exp, fade, hp, karplus, limiter, compressor, lp, make_ir,
                 mtof, noise, onepole, pan_st, pink, saw, sine, square)
from core import ROOT
from timeline import TIMELINE, TOTAL

RNG = np.random.default_rng(2024)
BPM = 96
BEAT = 60 / BPM
BAR = BEAT * 4
S16 = BEAT / 4

START = {type(sc).__name__: st for st, sc in TIMELINE}
SCN = {type(sc).__name__: sc for st, sc in TIMELINE}

CHORD = {
    "Dm": [50, 53, 57, 62], "Bb": [46, 50, 53, 58], "F": [48, 53, 57, 60], "C": [48, 52, 55, 60],
    "Gm": [43, 50, 55, 58], "A": [45, 49, 52, 57], "Dm9": [50, 53, 57, 64], "Fmaj7": [48, 52, 57, 60],
    "Bbmaj7": [46, 50, 53, 57], "Csus": [48, 50, 55, 60],
}
ROOT_ = {"Dm": 38, "Dm9": 38, "Bb": 34, "Bbmaj7": 34, "F": 41, "Fmaj7": 41, "C": 36, "Csus": 36, "Gm": 43, "A": 33}


def T(n):
    return np.arange(int(n * SR)) / SR


# =============================================================================== instruments

def braam(dur, notes=(26, 33, 38, 41, 45), bright=1.0, rel=None):
    n = int(dur * SR)
    t = T(dur)
    out = np.zeros(n)
    for m in notes:
        f = mtof(m)
        for det in (-0.13, 0.0, 0.11):
            fr = f * 2 ** (det / 12) * (1 + 0.003 * np.sin(2 * np.pi * (0.2 + RNG.random() * 0.3) * t))
            out += saw(fr, rng=RNG)
    out /= len(notes) * 1.8
    cut = 150 + 2200 * bright * np.exp(-t / 0.4) * np.minimum(t / 0.05, 1) + 350 * np.exp(-t / 2.0)
    y = lp(out, cut, 1.1)
    y = np.tanh(y * 2.2) * 0.8
    e = np.minimum(t / 0.025, 1) * np.exp(-t / (dur * 0.55))
    e *= fade(n, 0, rel if rel is not None else dur * 0.3)
    sub = np.sin(2 * np.pi * mtof(notes[0]) * t) * np.exp(-t / (dur * 0.5)) * 0.6
    return (y + sub * np.minimum(t / 0.02, 1)) * e


def boom(dur=3.0, f0=85, f1=30, drive=1.5):
    t = T(dur)
    f = f1 + (f0 - f1) * np.exp(-t / 0.18)
    y = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / (dur * 0.35)) * np.minimum(t / 0.002, 1)
    click = lp(noise(len(t), RNG), 2500) * np.exp(-t / 0.012) * 0.6
    return np.tanh((y + click) * drive) / np.tanh(drive)


def taiko(pitch=1.0, dur=1.2, vel=1.0):
    t = T(dur)
    f = (68 + 110 * np.exp(-t / 0.025)) * pitch
    body = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.42)
    skin = bp(noise(len(t), RNG), 260 * pitch, 1.3) * np.exp(-t / 0.10) * 0.9
    slap = hp(noise(len(t), RNG), 2000) * np.exp(-t / 0.006) * 0.25
    return np.tanh((body + skin + slap) * 1.6 * vel) * 0.8


def snare(dur=0.5, vel=1.0):
    t = T(dur)
    tone = np.sin(2 * np.pi * 190 * t) * np.exp(-t / 0.05)
    nz = bp(noise(len(t), RNG), 2500, 0.6) * np.exp(-t / 0.14)
    return (tone * 0.5 + nz) * vel


def crash(dur=4.0):
    t = T(dur)
    out = np.zeros((len(t), 2))
    for c in range(2):
        nz = hp(noise(len(t), RNG), 4000) * np.exp(-t / 1.6)
        met = sum(np.sin(2 * np.pi * f * t + RNG.random() * 6) for f in (3120, 4235, 5190, 6780, 7430, 8320)) * \
            np.exp(-t / 1.0) * 0.08
        out[:, c] = (nz * 0.5 + met) * np.minimum(t / 0.002, 1)
    return out


def riser(dur, f0=250, f1=9000, pitch=True, power=2.2):
    t = T(dur)
    u = t / dur
    fc = f0 * (f1 / f0) ** u
    nz = bp(noise(len(t), RNG), fc, 2.5) * 1.5
    out = nz
    if pitch:
        for base in (50, 62, 74):
            fr = mtof(base) * 2 ** (u * 2)
            out = out + lp(saw(fr, rng=RNG), 3000) * 0.18
    return out * u ** power


def rev_crash(dur):
    c = crash(dur + 0.2)[:int(dur * SR)][::-1]
    return c * (np.linspace(0, 1, c.shape[0]) ** 2 * fade(c.shape[0], 0, 0.008))[:, None]


def whoosh(dur=1.0, f0=300, f1=3500):
    t = T(dur)
    u = t / dur
    fc = f0 + (f1 - f0) * np.sin(np.pi * u)
    return bp(noise(len(t), RNG), fc, 1.2) * np.sin(np.pi * u) ** 2 * 1.4


def sub_drop(dur=3.0, f0=70, f1=24):
    t = T(dur)
    f = f1 + (f0 - f1) * np.exp(-t / (dur * 0.35))
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / (dur * 0.5)) * np.minimum(t / 0.01, 1)


def supersaw_note(m, dur, cut=1400, voices=5, spread=0.16, attack=1.2, release=1.5, stereo=True):
    n = int(dur * SR)
    t = T(dur)
    outs = []
    for c in range(2 if stereo else 1):
        acc = np.zeros(n)
        for v in range(voices):
            det = (v - (voices - 1) / 2) / ((voices - 1) / 2 + 1e-9) * spread
            acc += saw(np.full(n, mtof(m) * 2 ** (det / 12)), rng=RNG)
        acc /= voices
        lfo = 1 + 0.15 * np.sin(2 * np.pi * 0.13 * t + c)
        outs.append(lp(acc, cut * lfo, 0.8))
    e = np.minimum(t / attack, 1) * fade(n, 0, release)
    if stereo:
        return np.stack(outs, 1) * e[:, None]
    return outs[0] * e


def pad(dur, notes, cut=1300, level=1.0, attack=1.5, release=2.0, octave=0):
    out = np.zeros((int(dur * SR), 2))
    for m in notes:
        out += supersaw_note(m + octave, dur, cut, attack=attack, release=release)
    return out * level / max(1, len(notes)) * 1.2


VOWELS = {"a": ((800, 1.0), (1150, 0.5), (2900, 0.18)), "o": ((450, 1.0), (800, 0.45), (2830, 0.1)),
          "u": ((325, 1.0), (700, 0.3), (2530, 0.08)), "e": ((400, 1.0), (1700, 0.4), (2600, 0.2))}


def choir(dur, notes, vowel="a", level=1.0, attack=1.8, release=2.0):
    n = int(dur * SR)
    t = T(dur)
    out = np.zeros((n, 2))
    for m in notes:
        for c in range(2):
            src = np.zeros(n)
            for v in range(3):
                vib = 1 + 0.004 * np.sin(2 * np.pi * (5.2 + RNG.random()) * t + RNG.random() * 6)
                src += saw(mtof(m) * 2 ** ((v - 1) * 0.07 / 12) * vib, rng=RNG)
            y = np.zeros(n)
            for fc, g in VOWELS[vowel]:
                y += bp(src, fc, 7.0) * g
            out[:, c] += y
    e = np.minimum(t / attack, 1) * fade(n, 0, release)
    return out * e[:, None] * level / len(notes) * 2.2


def piano(m, dur=4.0, vel=0.8):
    t = T(dur)
    f = mtof(m)
    y = np.zeros(len(t))
    for k in range(1, 11):
        fk = f * k * math.sqrt(1 + 0.0004 * k * k)
        if fk > 16000:
            break
        y += np.sin(2 * np.pi * fk * t + RNG.random()) * (1 / k ** 1.25) * np.exp(-t / (2.8 / (1 + 0.45 * k)))
    ham = bp(noise(len(t), RNG), 2500, 1) * np.exp(-t / 0.01) * 0.1
    return (y * 0.5 + ham) * vel * np.minimum(t / 0.003, 1)


def bell(m, dur=4.0, vel=0.6):
    t = T(dur)
    f = mtof(m)
    y = np.zeros(len(t))
    for r, a, d in ((1, 1, 3.5), (2.0, 0.5, 2.2), (2.76, 0.45, 1.8), (5.4, 0.2, 0.9), (8.93, 0.1, 0.5)):
        y += np.sin(2 * np.pi * f * r * t) * a * np.exp(-t / d)
    return y * vel * 0.4 * np.minimum(t / 0.002, 1)


def church_bell(m, dur=6.0, vel=1.0):
    t = T(dur)
    f = mtof(m)
    y = np.zeros(len(t))
    for r, a, d in ((0.5, 0.6, 5), (1, 1, 4), (1.19, 0.6, 3), (1.5, 0.4, 2.5), (2.0, 0.5, 2), (2.51, 0.3, 1.5),
                    (3.0, 0.2, 1.0)):
        y += np.sin(2 * np.pi * f * r * t + RNG.random()) * a * np.exp(-t / d)
    return y * vel * 0.3 * np.minimum(t / 0.003, 1)


def guzheng(m, dur=3.0, vel=0.8, bend=0.0, bend_at=0.35):
    y = karplus(mtof(m), dur + 0.5, bright=0.75, decay=0.9975, rng=RNG)
    n = int(dur * SR)
    if bend:
        t = T(dur + 0.5)
        k = np.clip((t - bend_at) / 0.22, 0, 1)
        k = k * k * (3 - 2 * k)
        rate = 2 ** (bend * k / 12)
        idx = np.cumsum(rate)
        idx = idx[idx < y.size - 1]
        y = np.interp(idx, np.arange(y.size), y)
    y = y[:n] if y.size >= n else np.pad(y, (0, n - y.size))
    body = bp(y, 420, 2.0) * 0.4
    return (y + body) * vel * fade(n, 0.001, 0.3)


def theremin(events, total):
    """events: list of (t, midi, dur) relative; returns mono of length total."""
    n = int(total * SR)
    t = np.arange(n) / SR
    fcurve = np.full(n, mtof(events[0][1]))
    amp = np.zeros(n)
    for (t0, m, d) in events:
        i0, i1 = int(t0 * SR), int(min(total, t0 + d) * SR)
        fcurve[i0:] = mtof(m)
        amp[i0:i1] = 1
    fcurve = onepole(fcurve, 5.0)
    amp = onepole(amp, 4.0)
    vib = 1 + 0.006 * np.sin(2 * np.pi * 6.0 * t) * np.clip((t % 2.0) / 0.6, 0, 1)
    ph = np.cumsum(fcurve * vib) / SR
    y = np.sin(2 * np.pi * ph) + 0.18 * np.sin(4 * np.pi * ph) + 0.06 * np.sin(6 * np.pi * ph)
    return y * amp * 0.5


def pluck_saw(m, dur=0.3, cut=2500, vel=1.0):
    t = T(dur)
    y = saw(np.full(len(t), mtof(m)), rng=RNG) + saw(np.full(len(t), mtof(m) * 1.004), rng=RNG)
    y = lp(y, 300 + cut * np.exp(-t / 0.08), 1.2)
    return y * np.exp(-t / (dur * 0.35)) * np.minimum(t / 0.003, 1) * vel * 0.5


def string_note(m, dur=0.16, vel=1.0, cut=2200):
    t = T(dur + 0.05)
    y = saw(np.full(len(t), mtof(m)), rng=RNG) + 0.7 * saw(np.full(len(t), mtof(m) * 1.006), rng=RNG)
    y = lp(y, cut, 0.9)
    e = np.minimum(t / 0.006, 1) * np.exp(-t / (dur * 0.55))
    return np.tanh(y * e * 1.2) * vel * 0.45


def bass_note(m, dur=0.16, vel=1.0):
    t = T(dur + 0.05)
    y = saw(np.full(len(t), mtof(m)), rng=RNG) * 0.7 + np.sin(2 * np.pi * mtof(m) * t) * 0.6
    y = lp(y, 180 + 1400 * np.exp(-t / 0.05), 1.1)
    return np.tanh(y * 1.5) * np.minimum(t / 0.004, 1) * np.exp(-t / (dur * 0.6)) * vel * 0.6


def hat(dur=0.08, vel=1.0):
    t = T(dur)
    return hp(noise(len(t), RNG), 7500) * np.exp(-t / 0.025) * vel * 0.35


def tick(freq=3200, dur=0.03, vel=1.0):
    t = T(dur)
    return bp(noise(len(t), RNG), freq, 3) * np.exp(-t / 0.004) * vel


# =============================================================================== sound effects

def typewriter(vel=1.0):
    t = T(0.08)
    c = bp(noise(len(t), RNG), 2200 + RNG.random() * 800, 2) * np.exp(-t / 0.006)
    th = np.sin(2 * np.pi * 140 * t) * np.exp(-t / 0.015) * 0.5
    return (c + th) * vel


def stamp_sfx():
    t = T(0.6)
    thud = np.sin(2 * np.pi * (60 + 80 * np.exp(-t / 0.02)) * t) * np.exp(-t / 0.09)
    slap = bp(noise(len(t), RNG), 1400, 0.8) * np.exp(-t / 0.04)
    return np.tanh((thud * 1.4 + slap) * 1.5)


def paper_slap():
    t = T(0.4)
    return (bp(noise(len(t), RNG), 1600, 0.7) * np.exp(-t / 0.05) + np.sin(2 * np.pi * 90 * t) * np.exp(-t / 0.06)) \
        * 0.9


def shutter():
    t = T(0.25)
    y = np.zeros(len(t))
    for d in (0.0, 0.045):
        i = int(d * SR)
        seg = hp(noise(len(t) - i, RNG), 1500) * np.exp(-np.arange(len(t) - i) / SR / 0.004)
        y[i:] += seg
    pop = bp(noise(len(t), RNG), 900, 0.7) * np.exp(-t / 0.03) * 0.5
    return y * 0.7 + pop


def siren(dur):
    t = T(dur)
    f = 360 + 300 * (0.5 - 0.5 * np.cos(2 * np.pi * t / 7.0))
    y = lp(saw(f, rng=RNG), 1800) + 0.5 * np.sin(2 * np.pi * np.cumsum(f * 2) / SR)
    y = hp(lp(y, 1400), 250)
    return y * fade(len(t), 1.5, 2.0) * 0.35


def flak(vel=1.0):
    t = T(1.4)
    y = lp(noise(len(t), RNG), 260) * np.exp(-t / 0.35) * 2.5 + np.sin(2 * np.pi * 48 * t) * np.exp(-t / 0.3)
    return np.tanh(y * 1.2) * np.minimum(t / 0.002, 1) * vel


def radio_babble(dur, seed=0):
    rng = np.random.default_rng(seed)
    n = int(dur * SR)
    t = np.arange(n) / SR
    pitch = 120 * (1 + 0.12 * np.sin(2 * np.pi * 0.7 * t + rng.random() * 6) + 0.05 * np.sin(2 * np.pi * 2.3 * t))
    src = saw(pitch, rng=rng)
    gate = np.zeros(n)
    f1 = np.zeros(n)
    f2 = np.zeros(n)
    pos = 0.05
    vowels = [(700, 1200), (400, 2000), (300, 800), (550, 1700), (650, 1000), (350, 2300)]
    while pos < dur - 0.1:
        L = rng.uniform(0.07, 0.22)
        a, b = int(pos * SR), int(min(dur, pos + L) * SR)
        gate[a:b] = 1
        v = vowels[rng.integers(0, len(vowels))]
        f1[a:b] = v[0]
        f2[a:b] = v[1]
        pos += L + (rng.uniform(0.02, 0.09) if rng.random() < 0.8 else rng.uniform(0.2, 0.4))
    f1[f1 == 0] = 500
    f2[f2 == 0] = 1500
    f1 = onepole(f1, 25)
    f2 = onepole(f2, 25)
    gate = onepole(gate, 40)
    y = bp(src, f1, 6) + bp(src, f2, 8) * 0.6 + bp(noise(n, rng), 2500, 1) * 0.08
    y *= gate
    y = hp(lp(y, 2800), 380)
    y = np.tanh(y * 3) * 0.4
    st = bp(noise(n, rng), 3000, 0.7) * 0.05
    sq = np.zeros(n)
    for p in (0, n - int(0.05 * SR)):
        sq[p:p + int(0.05 * SR)] += bp(noise(int(0.05 * SR), rng), 3500, 1) * 0.5
    return (y + st + sq) * fade(n, 0.01, 0.05)


def static(dur, level=1.0, crackle=1.0):
    n = int(dur * SR)
    y = bp(noise(n, RNG), 2600, 0.5) * 0.25
    imp = np.zeros(n)
    k = int(dur * 60 * crackle)
    imp[RNG.integers(0, n, k)] = RNG.standard_normal(k) * 2
    y += hp(imp, 1500)
    return y * level


def radar_ping():
    t = T(1.2)
    return (np.sin(2 * np.pi * 1320 * t) + 0.25 * np.sin(2 * np.pi * 2640 * t)) * np.exp(-t / 0.18) * \
        np.minimum(t / 0.002, 1) * 0.5


def glitch_burst(dur=0.2):
    n = int(dur * SR)
    hold = int(SR / 900)
    y = np.repeat(RNG.standard_normal(n // hold + 1), hold)[:n]
    y += square(RNG.uniform(200, 1500), n) * 0.5
    return np.tanh(y * 1.5) * fade(n, 0.002, 0.03) * 0.45


def jet(dur):
    t = T(dur)
    u = t / dur
    y = lp(pink(len(t), RNG), 500 + 2500 * np.sin(np.pi * u) ** 2) * np.sin(np.pi * u) ** 1.5
    y += np.sin(2 * np.pi * np.cumsum(3800 - 1500 * u) / SR) * 0.03 * np.sin(np.pi * u)
    return y * 0.6


def prop_engine(dur):
    t = T(dur)
    y = lp(saw(np.full(len(t), 44.0), rng=RNG) * (1 + 0.5 * np.sin(2 * np.pi * 22 * t)), 500)
    y += lp(pink(len(t), RNG), 700) * 0.3
    return y * fade(len(t), 1.0, 1.0) * 0.35


def wind(dur, level=1.0):
    t = T(dur)
    y = bp(pink(len(t), RNG), 500 + 250 * np.sin(2 * np.pi * 0.11 * t), 0.6)
    return y * (0.6 + 0.4 * np.sin(2 * np.pi * 0.07 * t + 1)) * fade(len(t), 1.0, 1.0) * level


def heartbeat():
    t = T(0.8)
    a = np.sin(2 * np.pi * 55 * t) * np.exp(-t / 0.07)
    i = int(0.26 * SR)
    b = np.zeros(len(t))
    b[i:] = np.sin(2 * np.pi * 50 * t[:len(t) - i]) * np.exp(-t[:len(t) - i] / 0.07) * 0.7
    return np.tanh((a + b) * 2)


def wow_tone(dur=3.2):
    t = T(dur)
    env = np.exp(-((t - dur / 2) / (dur / 4.5)) ** 2)
    f = 780 + 6 * np.sin(2 * np.pi * 3 * t)
    y = np.sin(2 * np.pi * np.cumsum(f) / SR) + 0.3 * np.sin(2 * np.pi * np.cumsum(f * 1.5) / SR)
    return y * env * 0.5


def missile(dur):
    t = T(dur)
    u = t / dur
    y = bp(noise(len(t), RNG), 700 + 2500 * u, 1.5) * (0.3 + u) + lp(noise(len(t), RNG), 200) * 0.5
    return y * np.minimum(t / 0.05, 1) * 0.8


def flap_clatter(dur, rate=38):
    n = int(dur * SR)
    y = np.zeros(n)
    k = int(dur * rate)
    for i in range(k):
        p = int((i / rate + RNG.uniform(0, 0.01)) * SR)
        c = tick(RNG.uniform(1800, 4200), 0.03, RNG.uniform(0.5, 1.0))
        e = min(n, p + c.size)
        y[p:e] += c[:e - p]
    return y


def airport_hum(dur):
    t = T(dur)
    y = lp(pink(len(t), RNG), 900) * 0.4 + np.sin(2 * np.pi * np.cumsum(3100 + 50 * np.sin(2 * np.pi * 0.1 * t)) / SR) \
        * 0.02
    return y * fade(len(t), 1, 1)


# =============================================================================== mixer

class Mix:
    def __init__(self, dur):
        self.n = int(dur * SR)
        self.bus = {k: np.zeros((self.n, 2)) for k in ("mus", "sfx", "hall", "room", "dry")}

    def add(self, bus, t0, sig, pan=0.0, gain=1.0, send=None, amt=0.3):
        sig = np.asarray(sig, dtype=np.float64)
        st = pan_st(sig, pan) if sig.ndim == 1 else sig
        st = st * gain
        i0 = int(round(t0 * SR))
        if i0 < 0:
            st = st[-i0:]
            i0 = 0
        i1 = min(self.n, i0 + st.shape[0])
        if i1 <= i0:
            return
        self.bus[bus][i0:i1] += st[:i1 - i0]
        if send:
            self.bus[send][i0:i1] += st[:i1 - i0] * amt


M = Mix(TOTAL + 5)


def hit(t, size=1.0, notes=(26, 33, 38, 41, 45), braam_dur=5.0, crash_=True, taiko_=True, drop=True):
    M.add("mus", t, braam(braam_dur, notes), gain=0.55 * size, send="hall", amt=0.35)
    M.add("dry", t, boom(3.0), gain=0.75 * size)
    if drop:
        M.add("dry", t, sub_drop(3.5), gain=0.45 * size)
    if taiko_:
        for p, d in ((-0.4, 0.0), (0.4, 0.012)):
            M.add("mus", t + d, taiko(0.8, 1.6, 1.0), pan=p, gain=0.5 * size, send="hall", amt=0.4)
    if crash_:
        M.add("mus", t, crash(4.0), gain=0.22 * size, send="hall", amt=0.3)


def small_hit(t, size=1.0):
    M.add("mus", t, taiko(1.0, 1.2, 1.0), gain=0.55 * size, send="hall", amt=0.35)
    M.add("dry", t, boom(1.5, 70, 32), gain=0.45 * size)


def chord_pad(t0, chords, bar=BAR, cut=1300, level=0.3, choir_level=0.0, vowel="a", octave=0, attack=1.2):
    for i, ch in enumerate(chords):
        s = t0 + i * bar
        M.add("mus", s, pad(bar + 1.5, CHORD[ch], cut=cut, attack=attack, release=1.5, octave=octave), gain=level,
              send="hall", amt=0.45)
        if choir_level:
            M.add("mus", s, choir(bar + 1.5, [n + 12 for n in CHORD[ch][1:]], vowel, attack=attack), gain=choir_level,
                  send="hall", amt=0.6)


def drone(t0, dur, m=26, level=0.3, cut=260):
    n = int(dur * SR)
    tt = T(dur)
    y = np.zeros(n)
    for det in (-0.1, 0.0, 0.08):
        y += saw(np.full(n, mtof(m) * 2 ** (det / 12)), rng=RNG)
    y = lp(y / 3, cut * (1 + 0.3 * np.sin(2 * np.pi * 0.09 * tt)), 1.0)
    y += np.sin(2 * np.pi * mtof(m) * tt) * 0.6
    M.add("mus", t0, y * fade(n, 2.0, 2.5), gain=level, send="hall", amt=0.25)


OST = [0, 0, 12, 0, 0, 12, 0, 7, 0, 0, 12, 0, 7, 0, 12, 10]


def ostinato(t0, chords, level=0.25, bass=0.3, cut=2200, pattern=OST, octave=12):
    for i, ch in enumerate(chords):
        r = ROOT_[ch]
        for k in range(16):
            tt = t0 + i * BAR + k * S16
            acc = 1.0 if k % 4 == 0 else 0.7
            M.add("mus", tt, string_note(r + octave + pattern[k], 0.17, acc, cut), pan=0.25 * math.sin(k), gain=level,
                  send="hall", amt=0.2)
            if bass and k % 2 == 0:
                M.add("mus", tt, bass_note(r, 0.3, acc), gain=bass)


def groove(t0, bars, level=0.5, kick=(0, 6, 8, 11), snr=(4, 12), hats=True, fill_last=False):
    for b in range(bars):
        for k in range(16):
            tt = t0 + b * BAR + k * S16
            if k in kick:
                M.add("mus", tt, taiko(0.9 if k == 0 else 1.1, 1.0, 1.0 if k == 0 else 0.75), gain=level, send="room",
                      amt=0.3)
            if k in snr:
                M.add("mus", tt, snare(0.4, 0.8), gain=level * 0.55, pan=0.1, send="room", amt=0.35)
            if hats and k % 2 == 0:
                M.add("mus", tt, hat(0.08, 1.0 if k % 4 == 2 else 0.5), gain=level * 0.5, pan=0.3)
            if fill_last and b == bars - 1 and k >= 8:
                M.add("mus", tt, taiko(1.3 + 0.1 * (k - 8), 0.6, 0.7), pan=-0.3 + 0.08 * (k - 8), gain=level * 0.7,
                      send="room", amt=0.3)


def arp(t0, chords, level=0.2, inst="bell", octave=12, rate=S16, pattern=(0, 1, 2, 3, 2, 1, 0, 1)):
    for i, ch in enumerate(chords):
        notes = CHORD[ch]
        steps = int(BAR / rate)
        for k in range(steps):
            m = notes[pattern[k % len(pattern)] % len(notes)] + octave
            tt = t0 + i * BAR + k * rate
            sig = bell(m, 1.5, 0.5) if inst == "bell" else pluck_saw(m, 0.35, 2500, 0.8)
            M.add("mus", tt, sig, pan=0.5 * math.sin(k * 1.3), gain=level, send="hall", amt=0.5)


# =============================================================================== the cue sheet

def score():
    S = START

    # ---- 0–5 cold open: dead-air static, typing
    M.add("sfx", 0.0, static(5.0, 0.35) * fade(int(5.0 * SR), 0.5, 0.8), gain=0.5)
    drone(0.0, 6.0, 26, 0.25, 160)
    l1 = "本片内容取材自真实历史文献、政府解密档案与新闻报道"
    for i in range(len(l1)):
        M.add("sfx", 0.5 + i * 1.9 / len(l1), typewriter(0.5), pan=-0.3 + 0.6 * i / len(l1), gain=0.35, send="room",
              amt=0.2)
    for i in range(0, 64, 2):
        M.add("sfx", 2.1 + i * 1.6 / 64, tick(4200, 0.02, 0.4), gain=0.25)

    # ---- 5–12.5 FLIR teaser: pulse + radio chatter + riser, cut
    s = S["FlirTeaser"]
    for k in range(int(7.5 / (BEAT / 2))):
        tt = s + k * BEAT / 2
        M.add("mus", tt, bass_note(26, 0.25, 1.0 if k % 2 == 0 else 0.6), gain=0.55)
        if k % 2 == 1:
            M.add("mus", tt, tick(5000, 0.03, 0.6), gain=0.3, pan=0.4)
    M.add("mus", s, pad(7.6, [62, 63, 69], cut=900, attack=5.0, release=0.2), gain=0.18, send="hall", amt=0.3)
    M.add("sfx", s + 1.5, radio_babble(2.7, 1), gain=0.55, pan=-0.1)
    M.add("sfx", s + 4.4, radio_babble(2.8, 2), gain=0.55, pan=0.1)
    M.add("sfx", s, static(7.5, 0.25), gain=0.35)
    M.add("mus", s + 3.5, riser(4.0, 200, 8000), gain=0.35, send="hall", amt=0.2)
    M.add("sfx", s + 7.05, glitch_burst(0.45), gain=0.6)

    # ---- 12.5–20 hook
    s = S["Hook"]
    M.add("mus", s, rev_crash(0.5), gain=0.35)
    hit(s + 0.5, 1.2)
    drone(s + 0.5, 8.0, 26, 0.3, 200)
    for dt, m in ((1.3, 74), (1.9, 69), (2.5, 77), (3.1, 76), (4.4, 74), (5.0, 69), (5.6, 72), (6.2, 69)):
        M.add("mus", s + dt, piano(m, 4.0, 0.55), pan=0.2, gain=0.5, send="hall", amt=0.7)
    M.add("mus", s + 1.3, piano(38, 5.0, 0.6), gain=0.45, send="hall", amt=0.5)

    # ---- 20–30 Milky Way / chapter I
    s = S["MilkyWay"]
    chord_pad(s, ["Dm", "Dm", "Bb", "Bb"], level=0.28, choir_level=0.35, vowel="a", attack=2.0)
    small_hit(s, 0.9)
    M.add("mus", s, crash(5.0), gain=0.12, send="hall", amt=0.4)
    M.add("sfx", s + 4.2, whoosh(1.0, 800, 6000), pan=0.3, gain=0.25, send="hall", amt=0.4)
    for k in range(4):
        M.add("mus", s + k * BAR, taiko(0.7, 1.5, 0.8), gain=0.35, send="hall", amt=0.5)
    M.add("mus", s + 7.5, riser(2.5, 300, 5000, pitch=False), gain=0.2)

    # ---- 30–40 Plutarch: sky tears, object falls, impact at 35
    s = S["Plutarch"]
    chord_pad(s, ["Dm", "Dm", "Bb", "A"], level=0.22, choir_level=0.3, vowel="o")
    M.add("sfx", s + 0.3, boom(3.0, 120, 40), gain=0.5, send="hall", amt=0.4)
    M.add("sfx", s + 0.3, crash(3.0), gain=0.15, send="hall", amt=0.4)
    fall = 4.4
    tt = T(fall)
    roar = lp(brown(len(tt), RNG), 200 + 1800 * (tt / fall) ** 2) * (0.2 + (tt / fall) ** 2) * 0.8
    M.add("sfx", s + 0.6, roar, gain=0.6, send="hall", amt=0.3)
    M.add("sfx", s + 0.6, riser(4.4, 150, 3000, pitch=False, power=2.5), gain=0.4)
    for b in range(2):
        groove(s + b * BAR, 1, 0.35, kick=(0, 8, 10, 14), snr=(), hats=False)
    hit(s + 5.0, 1.4, braam_dur=5.0)
    M.add("sfx", s + 5.0, lp(brown(int(4 * SR), RNG), 300) * env_exp(int(4 * SR), 0.05, 1.5), gain=0.6)

    # ---- 40–52.5 Yangzhou pearl: guzheng over pad
    s = S["YangzhouPearl"]
    chord_pad(s, ["Dm", "Bb", "F", "C", "Dm"], level=0.2, choir_level=0.18, vowel="u", cut=1000)
    mel = [(0, 69, 1), (1, 74, 0.5), (1.5, 72, 0.5), (2, 69, 1), (3, 67, 1),
           (4, 65, 0.5), (4.5, 67, 0.5), (5, 69, 2, 2), (7, 62, 1),
           (8, 69, 0.5), (8.5, 72, 0.5), (9, 74, 1), (10, 77, 1), (11, 74, 1),
           (12, 72, 1), (13, 69, 0.5), (13.5, 67, 0.5), (14, 69, 2, 2),
           (16, 62, 1), (17, 57, 1), (18, 62, 3)]
    for ev in mel:
        b, m, d = ev[:3]
        bend = ev[3] if len(ev) > 3 else 0
        M.add("mus", s + b * BEAT, guzheng(m, max(2.0, d * BEAT + 1.2), 0.9, bend), pan=0.15, gain=0.55, send="hall",
              amt=0.45)
    for k, r in enumerate((50, 46, 53, 48, 50)):
        M.add("mus", s + k * BAR, guzheng(r - 12, 3.0, 0.8), pan=-0.2, gain=0.45, send="hall", amt=0.4)
    for k in range(6):
        M.add("mus", s + 1.2 + k * 0.28, bell(86 + (k % 3) * 3, 2.0, 0.3), pan=0.3, gain=0.25, send="hall", amt=0.6)
    M.add("mus", s + 3.3, riser(1.3, 1000, 9000, pitch=False, power=1.5), gain=0.12, send="hall", amt=0.5)
    M.add("mus", s + 4.6, bell(74, 4.0, 0.6), gain=0.35, send="hall", amt=0.7)
    M.add("mus", s + 4.6, bell(81, 4.0, 0.4), gain=0.3, send="hall", amt=0.7)
    M.add("sfx", s + 8.8, whoosh(2.0, 500, 5000), pan=0.6, gain=0.3, send="hall", amt=0.5)

    # ---- 52.5–62.5 Nuremberg: bells, dark choir, battle drums
    s = S["Nuremberg"]
    chord_pad(s, ["Gm", "Dm", "Bb", "A"], level=0.22, choir_level=0.4, vowel="a", cut=1100)
    for k in range(3):
        M.add("mus", s + k * 2.5, church_bell(50, 6.0, 1.0), gain=0.45, send="hall", amt=0.5)
    groove(s + 2 * BAR, 2, 0.35, kick=(0, 3, 6, 8, 11, 14), snr=(4, 12), hats=False, fill_last=True)
    M.add("mus", s + 7.5, riser(2.5, 200, 7000), gain=0.35)
    for k in range(6):
        M.add("sfx", s + 6.3 + k * 0.4, whoosh(0.7, 400, 2500), pan=RNG.uniform(-0.6, 0.6), gain=0.12)

    # ---- 62.5–125 the saucer age: ostinato section
    s = S["Airship"]
    hit(s, 1.0, braam_dur=3.5)
    prog = ["Dm", "Dm", "Bb", "Bb", "Gm", "Gm", "A", "A"]
    chords = (prog * 4)[:25]
    ostinato(s + BAR, chords[1:], level=0.16, bass=0.22)
    chord_pad(s, chords, level=0.12, cut=900)
    groove(s + BAR, 24, 0.28, kick=(0, 8, 10), snr=(), hats=False)
    M.add("sfx", s + 2.2, whoosh(1.0, 300, 4000), gain=0.35, send="room", amt=0.3)
    M.add("sfx", s + 3.2, paper_slap(), gain=0.6, send="room", amt=0.3)
    M.add("sfx", s + 5.6, whoosh(0.9, 300, 4000), gain=0.3, send="room", amt=0.3)
    M.add("sfx", s + 6.5, paper_slap(), gain=0.55, send="room", amt=0.3)

    s = S["LosAngeles1942"]
    small_hit(s, 1.0)
    M.add("sfx", s, siren(10.0), gain=0.35, pan=-0.3, send="hall", amt=0.5)
    la = SCN["LosAngeles1942"]
    la.ensure()
    for (bt, dx, dy, sz) in la.bursts:
        M.add("sfx", s + bt, flak(sz), pan=float(np.clip(dx / 400, -0.8, 0.8)), gain=0.3, send="hall", amt=0.4)
    groove(s, 4, 0.3, kick=(0, 3, 8, 11), snr=(4, 12), hats=True)

    s = S["Arnold1947"]
    small_hit(s, 0.8)
    M.add("sfx", s, prop_engine(10.0), gain=0.35)
    mel = [(0, 69, 2), (2, 74, 2), (4, 73, 1), (5, 69, 1), (6, 70, 4), (10, 69, 2), (12, 65, 2), (14, 64, 1), (15, 62, 1),
           (16, 62, 4), (20, 69, 2), (22, 74, 2), (24, 77, 2), (26, 76, 1), (27, 74, 1), (28, 73, 4)]
    th = theremin([(b * BEAT, m, d * BEAT) for b, m, d in mel], 20.5)
    M.add("mus", s, th, gain=0.32, pan=0.1, send="hall", amt=0.55)

    s = S["Roswell"]
    small_hit(s, 1.0)
    M.add("sfx", s, whoosh(1.0, 300, 5000), gain=0.35)
    M.add("sfx", s + 1.0, paper_slap(), gain=0.6, send="room", amt=0.3)
    for st_ in (4.6, 6.9):
        M.add("sfx", s + st_, stamp_sfx(), gain=0.8, send="room", amt=0.4)
        M.add("mus", s + st_, taiko(0.7, 1.2, 1.0), gain=0.45, send="hall", amt=0.3)

    s = S["BlueBook"]
    small_hit(s, 0.8)
    dur_typing = 420 / 58
    k = 0
    tt_ = 0.3
    while tt_ < 0.3 + min(dur_typing, 7.5):
        M.add("sfx", s + tt_, typewriter(RNG.uniform(0.6, 1.0)), pan=RNG.uniform(-0.3, 0.3), gain=0.45, send="room",
              amt=0.2)
        tt_ += RNG.uniform(0.055, 0.11)
        k += 1
    M.add("sfx", s + 5.4, stamp_sfx(), gain=0.9, send="room", amt=0.4)
    hit(s + 5.4, 0.7, braam_dur=3.0, crash_=False, drop=False)
    for i in range(40):
        M.add("sfx", s + 6.7 + i * 0.045, tick(2500, 0.02, 0.5), gain=0.3, pan=0.4)
    for i in range(30):
        M.add("sfx", s + 8.2 + i * 0.05, tick(1800, 0.02, 0.6), gain=0.35, pan=0.4)
    M.add("mus", s + 8.2, taiko(0.8, 1.2, 1.0), gain=0.4, send="hall", amt=0.3)

    s = S["Radar1952"]
    small_hit(s, 0.8)
    for k in range(3):
        M.add("sfx", s + 1.0 + k * 4.0, radar_ping(), gain=0.35, send="hall", amt=0.5)
    M.add("sfx", s, lp(pink(int(10 * SR), RNG), 300) * 0.3 + np.sin(2 * np.pi * 60 * T(10)) * 0.05, gain=0.4)
    for i in range(int(6.5 * 22)):
        M.add("sfx", s + 0.4 + i / 22 + RNG.uniform(0, 0.01), tick(3000, 0.02, 0.4), gain=0.2, pan=0.5)
    M.add("mus", s + 7.5, riser(2.5, 200, 9000), gain=0.4)
    M.add("mus", s + 9.5, rev_crash(0.5), gain=0.3)

    # ---- 125–147.5 SETI + Wow!: the breather
    s = S["Seti"]
    small_hit(s, 0.7)
    chord_pad(s, ["Fmaj7", "Dm9", "Bbmaj7", "Csus", "Fmaj7"], level=0.24, choir_level=0.22, vowel="o", attack=2.0)
    arp(s + BAR * 1, ["Dm9", "Bbmaj7", "Csus", "Fmaj7"], level=0.16, octave=12, rate=S16 * 2)
    for dt, m in ((0.4, 77), (2.9, 76), (5.4, 74), (7.9, 72), (10.4, 72)):
        M.add("mus", s + dt, piano(m, 4, 0.5), gain=0.4, pan=-0.2, send="hall", amt=0.7)
    M.add("sfx", s, wind(12.5, 0.15), gain=0.4)
    s = S["WowSignal"]
    chord_pad(s, ["Dm9", "Bbmaj7", "Csus", "Dm"], level=0.22, choir_level=0.2, vowel="u", attack=2.0)
    M.add("sfx", s, static(10.0, 0.35, 0.5), gain=0.35)
    M.add("sfx", s + 1.8, wow_tone(3.2), gain=0.45, send="hall", amt=0.5)
    M.add("mus", s + 3.4, bell(81, 3, 0.6), gain=0.3, send="hall", amt=0.6)
    M.add("mus", s + 7.5, riser(2.5, 200, 9000), gain=0.45)
    M.add("mus", s + 9.5, rev_crash(0.5), gain=0.35)

    # ---- 147.5–165 montage
    s = S["Montage"]
    hit(s, 1.1)
    chords = ["Dm", "Bb", "F", "C", "Dm", "Bb", "A"]
    ostinato(s, chords, level=0.2, bass=0.3, cut=2800)
    chord_pad(s, chords, level=0.14, choir_level=0.25, vowel="a")
    for k in range(6):
        c0 = s + 2.5 + k * 2.5
        M.add("sfx", c0, glitch_burst(0.18), gain=0.5)
        small_hit(c0, 1.0)
        M.add("mus", c0, braam(1.8, (38, 45, 50), 0.8), gain=0.25, send="hall", amt=0.3)
    groove(s + BAR, 6, 0.42, kick=(0, 3, 6, 8, 11, 14), snr=(4, 12), hats=True, fill_last=True)

    # ---- 165–175 Rendlesham: eerie
    s = S["Rendlesham"]
    drone(s, 11.0, 26, 0.35, 180)
    M.add("mus", s, choir(10.5, [62, 63, 69], "u", attack=3.0), gain=0.2, send="hall", amt=0.7)
    for k in range(int(10 * 0.875)):
        tt_ = s + k / 0.875
        M.add("mus", tt_, heartbeat(), gain=0.45)
    M.add("sfx", s, wind(10.0, 0.3), gain=0.5)
    M.add("sfx", s + 2.0, radio_babble(3.0, 5), gain=0.25, pan=-0.5)
    M.add("sfx", s, static(10.0, 0.2, 0.3), gain=0.25)

    # ---- 175–190 Belgium + Phoenix
    s = S["BelgianTriangle"]
    hit(s, 1.0, notes=(22, 29, 34, 41), braam_dur=6.0)
    chord_pad(s, ["Bb", "Bb", "Gm", "Gm", "Dm", "Dm"], level=0.18, choir_level=0.3, vowel="o", cut=900)
    M.add("sfx", s + 1.5, jet(5.5), pan=-0.5, gain=0.45, send="hall", amt=0.3)
    groove(s, 3, 0.3, kick=(0, 10), snr=(), hats=False)
    s = S["PhoenixLights"]
    small_hit(s, 0.8)
    M.add("mus", s, choir(7.6, [74, 77, 81], "a", attack=2.5), gain=0.25, send="hall", amt=0.7)
    arp(s, ["Gm", "Gm", "Dm"], level=0.12, octave=24, rate=S16 * 2)

    # ---- 190–200 Hangzhou
    s = S["Hangzhou"]
    small_hit(s, 1.0)
    M.add("sfx", s, airport_hum(10.0), gain=0.4)
    ostinato(s, ["Dm", "Dm", "Bb", "A"], level=0.14, bass=0.25, cut=1800)
    groove(s, 4, 0.3, kick=(0, 8, 10), snr=(4, 12), hats=True)
    M.add("sfx", s + 4.8, whoosh(0.4, 500, 5000), gain=0.3)
    M.add("sfx", s + 5.0, bell(84, 1.5, 0.6), gain=0.3, send="room", amt=0.3)
    M.add("sfx", s + 5.35, bell(79, 1.5, 0.6), gain=0.3, send="room", amt=0.3)
    for i in range(6):
        settle = 0.4 + i * 0.35
        M.add("sfx", s + 5.0 + settle - 0.35, flap_clatter(0.35, 40), pan=0.3, gain=0.35)
    M.add("mus", s + 7.5, riser(2.5, 200, 9000), gain=0.4)

    # ---- 200–222.5 Navy FLIR + news: hybrid pulse
    s = S["NavyFlir"]
    hit(s, 1.2)
    chords = ["Dm", "Dm", "Bb", "Bb", "F", "F", "C", "C", "A"]
    for i, ch in enumerate(chords):
        for k in range(16):
            tt_ = s + BAR + i * BAR + k * S16
            if tt_ > S["Hearing"]:
                break
            M.add("mus", tt_, bass_note(ROOT_[ch], 0.14, 1.0 if k % 4 == 0 else 0.55), gain=0.4)
    ostinato(s + BAR, chords[:8], level=0.13, bass=0, cut=3000)
    groove(s + BAR, 8, 0.35, kick=(0, 8), snr=(4, 12), hats=True)
    M.add("sfx", s + 5.0, glitch_burst(0.2), gain=0.5)
    M.add("sfx", s + 8.75, glitch_burst(0.2), gain=0.5)
    M.add("sfx", s + 5.1, radio_babble(1.8, 11), gain=0.5)
    M.add("sfx", s + 6.9, radio_babble(1.8, 12), gain=0.5)
    M.add("sfx", s + 8.8, radio_babble(1.8, 13), gain=0.5)
    M.add("sfx", s + 10.6, radio_babble(1.9, 14), gain=0.5)
    M.add("sfx", s, static(12.5, 0.2), gain=0.3)
    s = S["NewsMontage"]
    for k in range(3):
        c0 = s + k * 3.34
        hit(c0, 0.75, notes=(38, 45, 50, 53), braam_dur=2.5, crash_=True, drop=False)
        M.add("sfx", c0, glitch_burst(0.15), gain=0.4)
        for i in range(8):
            M.add("sfx", c0 + 0.3 + i * BEAT / 2, tick(2800, 0.02, 0.6), gain=0.25, pan=-0.4)

    # ---- 222.5–235 hearing
    s = S["Hearing"]
    drone(s, 13.0, 26, 0.35, 220)
    chord_pad(s, ["Dm", "Dm", "Bb", "Gm", "Gm"], level=0.14, choir_level=0.18, vowel="u", cut=800)
    hr = SCN["Hearing"]
    hr.ensure()
    for ft in hr.flashes:
        M.add("sfx", s + ft, shutter(), pan=RNG.uniform(-0.7, 0.7), gain=0.3, send="room", amt=0.4)
    M.add("mus", s + 5.4, rev_crash(0.5), gain=0.35)
    hit(s + 5.9, 1.2, braam_dur=4.5)
    for dt, m in ((9.1, 74), (9.7, 69), (10.3, 77), (10.9, 76)):
        M.add("mus", s + dt, piano(m, 4.0, 0.5), gain=0.45, send="hall", amt=0.7)

    # ---- 235–242.5 Hellfire
    s = S["Hellfire"]
    M.add("mus", s, pad(7.6, [45, 49, 52, 57], cut=900, attack=3.0), gain=0.2, send="hall", amt=0.4)
    for k in range(int(4.0 / (BEAT / 2))):
        M.add("mus", s + k * BEAT / 2, tick(4000, 0.03, 0.5 + 0.5 * k / 12), gain=0.3)
    M.add("mus", s + 1.5, riser(2.5, 200, 6000), gain=0.3)
    M.add("sfx", s + 2.9, missile(1.1), gain=0.6, pan=0.5)
    M.add("sfx", s + 4.0, boom(3.0, 110, 35, 2.0), gain=0.85)
    hit(s + 4.0, 1.0, crash_=True)
    M.add("sfx", s + 4.0, static(3.5, 0.5) * np.linspace(1, 0, int(3.5 * SR)), gain=0.4)

    # ---- 242.5–260 visitors from the stars: awe build
    s = S["Interstellar"]
    small_hit(s, 0.9)
    chords = ["F", "C", "Dm", "Bb", "F", "C", "Dm"]
    chord_pad(s, chords, level=0.2, choir_level=0.3, vowel="a", cut=1600)
    arp(s + BAR, chords[1:], level=0.16, inst="pluck", octave=12)
    groove(s + 2 * BAR, 5, 0.3, kick=(0, 8), snr=(), hats=True)
    groove(s + 5 * BAR, 2, 0.4, kick=(0, 6, 8, 11, 14), snr=(4, 12), hats=True)
    M.add("sfx", s + 2.2, whoosh(1.4, 300, 4000), pan=0.3, gain=0.3, send="hall", amt=0.4)
    M.add("sfx", s + 5.2, whoosh(1.4, 300, 4000), pan=-0.3, gain=0.3, send="hall", amt=0.4)
    small_hit(S["Mars"], 1.0)

    # ---- 260–267.5 recap: accelerating hits, riser, silence
    s = S["Recap"]
    for k in range(16):
        tt_ = s + k * 0.3125
        big = k % 4 == 0
        M.add("mus", tt_, taiko(1.0 if big else 1.3, 0.8, 1.0 if big else 0.7), pan=0.3 * math.sin(k), gain=0.5,
              send="hall", amt=0.3)
        M.add("sfx", tt_, glitch_burst(0.08), gain=0.2)
        if big:
            M.add("mus", tt_, braam(1.3, (38, 45, 50, 53), 1.0), gain=0.35, send="hall", amt=0.3)
            M.add("dry", tt_, boom(1.2, 80, 34), gain=0.5)
    riz = riser(7.0, 150, 12000, power=2.0)
    M.add("mus", s, riz, gain=0.5, send="hall", amt=0.2)
    M.add("mus", s + 5.0, pad(2.0, [50, 57, 62, 69], cut=3000, attack=1.8, release=0.02), gain=0.25)
    M.add("mus", s + 6.3, rev_crash(0.7), gain=0.5)

    # ---- 267.5 TITLE
    s = S["Title"]
    hit(s, 1.6, notes=(26, 33, 38, 41, 45, 50), braam_dur=7.0)
    M.add("mus", s, choir(9.5, [62, 65, 69, 74], "a", attack=0.4, release=3.0), gain=0.4, send="hall", amt=0.6)
    M.add("mus", s, pad(9.5, [50, 57, 62, 65], cut=1800, attack=0.3, release=3.0), gain=0.3, send="hall", amt=0.5)
    M.add("mus", s + 2.5, riser(2.5, 1000, 12000, pitch=False, power=1.5), gain=0.15, send="hall", amt=0.5)
    M.add("mus", s + 4.8, bell(74, 5, 0.8), gain=0.4, send="hall", amt=0.7)
    hit(s + 4.8, 0.8, notes=(22, 29, 34, 41, 46), braam_dur=5.0, drop=False)
    M.add("mus", s + 4.8, choir(5.5, [58, 62, 65, 70], "o", attack=0.8, release=2.5), gain=0.35, send="hall", amt=0.6)

    # ---- 277.5–287.5 end card
    s = S["EndCard"]
    M.add("mus", s, pad(10.0, [50, 57, 62, 65], cut=900, attack=2.0, release=5.0), gain=0.2, send="hall", amt=0.6)
    for dt, m in ((0.5, 74), (1.1, 69), (1.7, 77), (2.3, 76), (3.6, 74), (4.2, 69), (4.8, 72), (5.4, 69), (7.0, 62)):
        M.add("mus", s + dt, piano(m, 5.0, 0.5), gain=0.45, send="hall", amt=0.75)
    M.add("mus", s + 0.5, piano(38, 6.0, 0.5), gain=0.4, send="hall", amt=0.6)
    M.add("sfx", s + 8.45, whoosh(0.35, 1500, 9000), pan=0.7, gain=0.25, send="hall", amt=0.6)
    M.add("sfx", s + 8.0, static(2.0, 0.2) * fade(int(2 * SR), 0.5, 1.0), gain=0.25)


AUTO = [(0, 0.8), (12.4, 0.9), (13.0, 1.1), (19.5, 0.85), (30, 0.95), (35, 1.1), (39.5, 0.8), (52.5, 0.9),
        (62.5, 1.0), (122, 1.0), (125, 0.7), (147.3, 0.72), (147.5, 1.1), (164.8, 1.05), (165, 0.75), (174.8, 0.8),
        (175, 1.05), (190, 0.9), (200, 1.1), (222.5, 0.9), (228.4, 1.1), (231, 0.8), (239, 1.1), (242.5, 0.8),
        (256, 0.95), (260, 1.0), (267.4, 1.0), (267.5, 1.35), (276, 1.1), (277.5, 0.8), (TOTAL + 5, 0.7)]


def automation(n):
    t = np.arange(n) / SR
    k = np.array(AUTO)
    return np.interp(t, k[:, 0], k[:, 1])


def gate(n):
    t = np.arange(n) / SR
    g = np.ones(n)
    for a, b, depth in ((12.47, 12.98, 0.12), (34.8, 34.99, 0.2), (266.96, 267.49, 0.0)):
        m = (t > a) & (t < b)
        g[m] = depth
        ra = (t > a - 0.04) & (t <= a)
        g[ra] = 1 - (1 - depth) * (t[ra] - (a - 0.04)) / 0.04
    return onepole(g, 150)


def render():
    score()
    print("mixing...")
    hall = convolve_st(M.bus["hall"], make_ir(3.8, 4.5, 0.03, 5500, 1400, seed=1))
    room = convolve_st(M.bus["room"], make_ir(1.1, 1.4, 0.01, 7000, 2500, seed=2))
    auto = automation(M.n)[:, None]
    music = (M.bus["mus"] * 0.9 + hall * 0.55) * auto
    music = compressor(music, -24, 1.8, 0.02, 0.3, 0.0)
    fx = M.bus["sfx"] + room * 0.5
    mix = music + fx * 0.9 + M.bus["dry"] * 0.9 * np.sqrt(auto)
    mix = np.stack([hp(mix[:, 0], 22), hp(mix[:, 1], 22)], 1)
    mix = compressor(mix, -12, 1.6, 0.01, 0.25, 0.0)
    rms = np.sqrt(np.mean(mix[int(20 * SR):int(260 * SR)] ** 2))
    mix *= 0.15 / rms
    mix *= gate(M.n)[:, None]
    mix = limiter(mix, 0.89)
    n = int(TOTAL * SR)
    mix = mix[:n]
    mix[-int(1.0 * SR):] *= np.linspace(1, 0, int(1.0 * SR))[:, None]
    out = os.path.join(ROOT, "build", "score.wav")
    wavfile.write(out, SR, (np.clip(mix, -1, 1) * 32767).astype(np.int16))
    print("wrote", out, "peak", np.abs(mix).max(), "rms", np.sqrt(np.mean(mix ** 2)))


if __name__ == "__main__":
    render()
