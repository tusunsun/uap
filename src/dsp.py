"""Small DSP toolkit (numba accelerated) used by the score generator."""
import math

import numpy as np
from numba import njit
from scipy.ndimage import minimum_filter1d
from scipy.signal import oaconvolve

SR = 48000


def secs(n):
    return np.arange(int(n * SR), dtype=np.float64) / SR


def mtof(m):
    return 440.0 * 2.0 ** ((np.asarray(m, dtype=np.float64) - 69) / 12.0)


@njit(cache=True)
def _svf(x, fc, q, mode, sr):
    y = np.empty_like(x)
    ic1 = 0.0
    ic2 = 0.0
    k = 1.0 / q
    for n in range(x.shape[0]):
        f = fc[n]
        if f > sr * 0.45:
            f = sr * 0.45
        if f < 10.0:
            f = 10.0
        g = math.tan(math.pi * f / sr)
        a1 = 1.0 / (1.0 + g * (g + k))
        a2 = g * a1
        a3 = g * a2
        v3 = x[n] - ic2
        v1 = a1 * ic1 + a2 * v3
        v2 = ic2 + a2 * ic1 + a3 * v3
        ic1 = 2.0 * v1 - ic1
        ic2 = 2.0 * v2 - ic2
        if mode == 0:
            y[n] = v2
        elif mode == 1:
            y[n] = v1 * k
        else:
            y[n] = x[n] - k * v1 - v2
    return y


def svf(x, fc, q=0.707, mode="lp"):
    x = np.ascontiguousarray(x, dtype=np.float64)
    fc = np.broadcast_to(np.asarray(fc, dtype=np.float64), x.shape).copy()
    return _svf(x, fc, float(q), {"lp": 0, "bp": 1, "hp": 2}[mode], float(SR))


def lp(x, fc, q=0.707):
    return svf(x, fc, q, "lp")


def hp(x, fc, q=0.707):
    return svf(x, fc, q, "hp")


def bp(x, fc, q=1.0):
    return svf(x, fc, q, "bp")


@njit(cache=True)
def _saw(freq, ph, sr):
    out = np.empty_like(freq)
    for n in range(freq.shape[0]):
        dt = freq[n] / sr
        v = 2.0 * ph - 1.0
        if ph < dt:
            t = ph / dt
            v -= t + t - t * t - 1.0
        elif ph > 1.0 - dt:
            t = (ph - 1.0) / dt
            v -= t * t + t + t + 1.0
        out[n] = v
        ph += dt
        if ph >= 1.0:
            ph -= 1.0
    return out


def saw(freq, n=None, phase=None, rng=None):
    freq = np.asarray(freq, dtype=np.float64)
    if freq.ndim == 0:
        freq = np.full(n, float(freq))
    if phase is None:
        phase = (rng or np.random).random()
    return _saw(np.ascontiguousarray(freq), float(phase), float(SR))


def sine(freq, n=None, phase=0.0):
    freq = np.asarray(freq, dtype=np.float64)
    if freq.ndim == 0:
        return np.sin(2 * np.pi * freq * np.arange(n) / SR + phase)
    ph = np.cumsum(freq) / SR
    return np.sin(2 * np.pi * ph + phase)


def square(freq, n=None, duty=0.5):
    freq = np.asarray(freq, dtype=np.float64)
    if freq.ndim == 0:
        freq = np.full(n, float(freq))
    a = saw(freq, phase=0.0)
    b = saw(freq, phase=duty % 1.0)
    return (a - b) * 0.5


@njit(cache=True)
def _onepole(x, a):
    y = np.empty_like(x)
    s = 0.0
    for n in range(x.shape[0]):
        s += a * (x[n] - s)
        y[n] = s
    return y


def onepole(x, fc):
    a = 1 - math.exp(-2 * math.pi * fc / SR)
    return _onepole(np.ascontiguousarray(x, dtype=np.float64), a)


def noise(n, rng=None):
    rng = rng or np.random.default_rng()
    return rng.standard_normal(n)


def pink(n, rng=None):
    rng = rng or np.random.default_rng()
    w = rng.standard_normal(n)
    X = np.fft.rfft(w)
    f = np.arange(X.size)
    f[0] = 1
    X /= np.sqrt(f)
    y = np.fft.irfft(X, n)
    return y / (np.std(y) + 1e-9)


def brown(n, rng=None):
    y = np.cumsum((rng or np.random.default_rng()).standard_normal(n))
    y = hp(y, 20)
    return y / (np.std(y) + 1e-9)


def env_adsr(n, a, d, s, r, total=None):
    """sample-count envelope; a,d,r in seconds; s level; note held for `total` s."""
    total = total if total is not None else n / SR
    t = np.arange(n) / SR
    e = np.where(t < a, t / max(a, 1e-4), s + (1 - s) * np.exp(-(t - a) / max(d, 1e-4)))
    rel = np.clip(1 - (t - total) / max(r, 1e-4), 0, 1)
    return e * np.where(t > total, rel, 1.0)


def env_exp(n, attack, decay):
    t = np.arange(n) / SR
    return np.minimum(t / max(attack, 1e-4), 1.0) * np.exp(-t / decay)


def fade(n, fin, fout):
    e = np.ones(n)
    a = int(fin * SR)
    b = int(fout * SR)
    if a > 0:
        e[:a] = np.linspace(0, 1, a) ** 2
    if b > 0:
        e[-b:] *= np.linspace(1, 0, b) ** 2
    return e


@njit(cache=True)
def _ks(n, L, seed_buf, decay, damp):
    buf = seed_buf.copy()
    out = np.empty(n)
    i = 0
    prev = 0.0
    for k in range(n):
        v = buf[i]
        out[k] = v
        j = i + 1
        if j >= L:
            j = 0
        nv = decay * ((1 - damp) * v + damp * buf[j])
        buf[i] = nv
        i = j
    return out


def karplus(freq, dur, bright=0.5, decay=0.996, rng=None):
    rng = rng or np.random.default_rng()
    Lf = SR / freq
    L = int(Lf)
    exc = rng.standard_normal(L)
    exc = onepole(exc, 800 + 6000 * bright)
    exc -= exc.mean()
    n = int(dur * SR)
    y = _ks(n + 10, L, exc, decay, 0.5)
    # integer delay plays slightly sharp: read it back slower by L/Lf to land on pitch
    idx = np.arange(n) * (L / Lf)
    return np.interp(idx, np.arange(y.size), y)


def pan_st(x, pan=0.0):
    a = (pan + 1) * math.pi / 4
    return np.stack([x * math.cos(a), x * math.sin(a)], 1)


def make_ir(t60=3.0, length=None, predelay=0.02, bright=6000, dark=1200, seed=0, width=1.0):
    rng = np.random.default_rng(seed)
    length = length or t60 * 1.2
    n = int(length * SR)
    t = np.arange(n) / SR
    ir = np.zeros((n, 2))
    for c in range(2):
        w = rng.standard_normal(n)
        a = lp(w, bright)
        b = lp(w, dark)
        mixk = np.clip(t / (t60 * 0.6), 0, 1)
        v = a * (1 - mixk) + b * mixk
        v *= np.exp(-6.9 * t / t60)
        ir[:, c] = v
    pd = int(predelay * SR)
    ir = np.concatenate([np.zeros((pd, 2)), ir])[:n]
    # early reflections
    for k in range(10):
        d = int(rng.uniform(0.008, 0.08) * SR) + pd
        g = rng.uniform(0.2, 0.6) * (1 - k / 12)
        ir[d, 0] += g * rng.choice([-1, 1])
        ir[d + int(rng.uniform(0, 0.004) * SR), 1] += g * rng.choice([-1, 1])
    mid = (ir[:, 0] + ir[:, 1]) / 2
    side = (ir[:, 0] - ir[:, 1]) / 2 * width
    ir = np.stack([mid + side, mid - side], 1)
    ir /= np.sqrt((ir ** 2).sum() / 2)
    return ir


def convolve_st(x, ir):
    return np.stack([oaconvolve(x[:, 0], ir[:, 0])[:x.shape[0]], oaconvolve(x[:, 1], ir[:, 1])[:x.shape[0]]], 1)


@njit(cache=True)
def _smooth_gain(g, att, rel):
    y = np.empty_like(g)
    s = 1.0
    for n in range(g.shape[0]):
        v = g[n]
        if v < s:
            s += att * (v - s)
        else:
            s += rel * (v - s)
        y[n] = s
    return y


def limiter(x, ceiling=0.89, lookahead=0.004, release=0.12):
    peak = np.max(np.abs(x), axis=1)
    need = np.minimum(1.0, ceiling / np.maximum(peak, 1e-9))
    la = int(lookahead * SR)
    need = minimum_filter1d(need, size=2 * la + 1, origin=0)
    need = np.concatenate([need[la:], np.ones(la)])
    g = _smooth_gain(need, 1 - math.exp(-1 / (0.0015 * SR)), 1 - math.exp(-1 / (release * SR)))
    g = np.minimum(g, need)
    return x * g[:, None]


def compressor(x, thresh_db=-18, ratio=3.0, attack=0.01, release=0.2, makeup_db=0.0):
    lvl = np.sqrt(np.convolve((x ** 2).mean(1), np.ones(int(0.01 * SR)) / int(0.01 * SR), mode="same")) + 1e-9
    db = 20 * np.log10(lvl)
    over = np.maximum(db - thresh_db, 0)
    gr_db = -over * (1 - 1 / ratio)
    g = 10 ** (gr_db / 20)
    g = _smooth_gain(g, 1 - math.exp(-1 / (attack * SR)), 1 - math.exp(-1 / (release * SR)))
    return x * (g * 10 ** (makeup_db / 20))[:, None]
