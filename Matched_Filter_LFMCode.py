import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Utilities you already have
# ----------------------------
def normalize_energy(x: np.ndarray):
    e = np.sum(np.abs(x) ** 2) + 1e-12
    return x / np.sqrt(e)

def apply_delay(x: np.ndarray, delay_samp: int):
    if delay_samp <= 0:
        return x
    return np.concatenate([np.zeros(delay_samp, dtype=x.dtype), x])

def apply_doppler(x: np.ndarray, fd_hz: float, fs: float):
    if fd_hz == 0.0:
        return x
    n = np.arange(len(x))
    return x * np.exp(1j * 2 * np.pi * fd_hz * n / fs)

def add_awgn(x: np.ndarray, snr_db: float):
    sig_power = np.mean(np.abs(x) ** 2)
    snr_lin = 10 ** (snr_db / 10)
    noise_power = sig_power / snr_lin
    noise = np.sqrt(noise_power / 2) * (np.random.randn(*x.shape) + 1j * np.random.randn(*x.shape))
    return x + noise

def matched_filter(rx: np.ndarray, ref: np.ndarray):
    h = ref[::-1].conj()
    return np.convolve(rx, h, mode="full")

def peak_and_pslr_db(y: np.ndarray, guard: int = 5):
    mag = np.abs(y)
    peak = float(mag.max())
    peak_idx = int(mag.argmax())

    mask = np.ones_like(mag, dtype=bool)
    lo = max(0, peak_idx - guard)
    hi = min(len(mag), peak_idx + guard + 1)
    mask[lo:hi] = False
    sidelobe = float(mag[mask].max()) if np.any(mask) else 0.0

    pslr_db = 20 * np.log10((peak + 1e-12) / (sidelobe + 1e-12))
    return peak, peak_idx, pslr_db

# ----------------------------
# LFM generator (complex baseband)
# ----------------------------
def make_lfm(T: float, B: float, fs: float, f1: float = None, f2: float = None):
    """
    Create complex baseband LFM chirp of duration T, bandwidth B, sampling fs.
    If f1/f2 not specified: sweep from -B/2 to +B/2 (centered at 0 Hz).
    """
    N = int(np.round(T * fs))
    t = np.arange(N) / fs

    if f1 is None or f2 is None:
        f1 = -B / 2
        f2 = +B / 2

    k = (f2 - f1) / T  # Hz/s
    phase = 2 * np.pi * (f1 * t + 0.5 * k * t**2)
    x = np.exp(1j * phase).astype(np.complex128)
    return x

# ----------------------------
# Partial DRFM capture model
# ----------------------------
def make_drfm_partial(ref: np.ndarray, fs: float, cap_start_s: float, cap_dur_s: float,
                      replay_mode: str = "same_time"):
    """
    replay_mode:
      - "same_time": captured segment stays at same time indices (missing parts are zero)
      - "from_start": captured segment is replayed starting at t=0 (rest zero)
    """
    N = len(ref)
    i0 = int(np.round(cap_start_s * fs))
    i1 = int(np.round((cap_start_s + cap_dur_s) * fs))
    i0 = np.clip(i0, 0, N)
    i1 = np.clip(i1, 0, N)
    if i1 <= i0:
        return np.zeros_like(ref)

    seg = ref[i0:i1]

    y = np.zeros_like(ref)
    if replay_mode == "same_time":
        y[i0:i1] = seg
    elif replay_mode == "from_start":
        L = len(seg)
        y[0:L] = seg
    else:
        raise ValueError("replay_mode must be 'same_time' or 'from_start'")
    return y

# ----------------------------
# MAIN demo (parameter-selectable)
# ----------------------------
if __name__ == "__main__":
    # ---- LFM params (your selectable knobs) ----
    fs         = 100e6        # Hz
    T_pulse    = 50e-6        # 50 us
    B          = 60e6         # 60 MHz sweep
    f1, f2     = -B/2, +B/2   # baseband-centered chirp; set explicitly if you want

    # ---- DRFM partial capture params ----
    cap_start  = 0e-6         # seconds into the pulse (e.g., 0 us)
    cap_dur    = 20e-6        # capture duration (e.g., 30 us of 50 us)
    replay_mode = "same_time" # "same_time" or "from_start"

    # ---- Channel params ----
    delay_samp = 400
    fd_hz      = 1e3
    snr_db     = 30.0

    # ---- Build reference (radar known transmit) ----
    ref = make_lfm(T_pulse, B, fs, f1=f1, f2=f2)
    ref = normalize_energy(ref)

    # ---- Build uncoded pulse (same length) ----
    normal = np.ones_like(ref, dtype=np.complex128)
    normal = normalize_energy(normal)

    # ---- Build partial DRFM echo ----
    drfm_partial = make_drfm_partial(ref, fs, cap_start, cap_dur, replay_mode=replay_mode)
    drfm_partial = normalize_energy(drfm_partial)  # optional: keep energy comparable

    # ---- Simulate received echoes ----
    rx_ref = add_awgn(apply_doppler(apply_delay(ref, delay_samp), fd_hz, fs), snr_db)
    rx_nrm = add_awgn(apply_doppler(apply_delay(normal, delay_samp), fd_hz, fs), snr_db)
    rx_drm = add_awgn(apply_doppler(apply_delay(drfm_partial, delay_samp), fd_hz, fs), snr_db)

    # ---- Matched filter using LFM reference ----
    y_ref = matched_filter(rx_ref, ref)
    y_nrm = matched_filter(rx_nrm, ref)
    y_drm = matched_filter(rx_drm, ref)

    peak_r, idx_r, pslr_r = peak_and_pslr_db(y_ref)
    peak_n, idx_n, pslr_n = peak_and_pslr_db(y_nrm)
    peak_d, idx_d, pslr_d = peak_and_pslr_db(y_drm)

    print("\n=== LFM MATCHED FILTER TEST ===")
    print(f"fs={fs/1e6:.1f} MHz, T={T_pulse*1e6:.1f} us, B={B/1e6:.1f} MHz")
    print(f"DRFM capture: start={cap_start*1e6:.1f} us, dur={cap_dur*1e6:.1f} us, mode={replay_mode}")
    print(f"SNR={snr_db} dB, fd={fd_hz} Hz, delay={delay_samp} samp")

    ratio_dn_db = 20*np.log10((peak_r+1e-12)/(peak_n+1e-12))
    ratio_dd_db = 20*np.log10((peak_r+1e-12)/(peak_d+1e-12))
    print("\nCase A: Echo IS LFM (matched)")
    print(f"  Peak={peak_r:.6f}, PSLR={pslr_r:.2f} dB")
    print("\nCase B: Echo is NORMAL pulse (mismatched)")
    print(f"  Peak={peak_n:.6f}, PSLR={pslr_n:.2f} dB")
    print("\nCase C: Echo is DRFM partial-capture (mismatched)")
    print(f"  Peak={peak_d:.6f}, PSLR={pslr_d:.2f} dB")
    print("\nDiscrimination:")
    print(f"  Coded/Normal peak ratio: {ratio_dn_db:.2f} dB")
    print(f"  Coded/DRFM   peak ratio: {ratio_dd_db:.2f} dB")

    # ---- Plot MF outputs in dB ----
    yref_db = 20*np.log10(np.abs(y_ref)+1e-12)
    ynrm_db = 20*np.log10(np.abs(y_nrm)+1e-12)
    ydrm_db = 20*np.log10(np.abs(y_drm)+1e-12)

    plt.figure(figsize=(12,5))
    plt.plot(yref_db, label="Echo = LFM (matched)")
    plt.plot(ynrm_db, label="Echo = Normal pulse")
    plt.plot(ydrm_db, label="Echo = DRFM partial")
    plt.grid(True)
    plt.ylim(-60, 5)
    plt.xlabel("Matched filter output index")
    plt.ylabel("|y| (dB)")
    plt.title("LFM matched filter comparison")
    plt.legend()
    plt.show()
