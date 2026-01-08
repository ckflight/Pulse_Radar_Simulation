import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# make_phase_code generates 64 phase values which will be put onto the pulse
# ----------------------------
def make_phase_code(code_type: str, N: int):
    """
    Real-radar-oriented phase codes (unit magnitude complex sequence).

    Supported code_type:
      - "barker" or "barker13"/"barker11"/...  (BPSK ±1)
      - "frank"  (polyphase, requires N = M^2)
      - "p1"|"p2"|"p3"|"p4"  (polyphase, requires N = M^2)

    Returns:
      code: np.ndarray, dtype=complex128, length N (or Barker length if "barker13" used)
    """

    code_type = code_type.lower().strip()

    # ----------------------------
    # Barker codes (binary phase)
    # ----------------------------
    barker_table = {
        2:  [ 1, -1],
        3:  [ 1,  1, -1],
        4:  [ 1,  1, -1,  1],
        5:  [ 1,  1,  1, -1,  1],
        7:  [ 1,  1,  1, -1, -1,  1, -1],
        11: [ 1,  1,  1, -1, -1, -1,  1, -1, -1,  1, -1],
        13: [ 1,  1,  1,  1,  1, -1, -1,  1,  1, -1,  1, -1,  1],
    }

    if code_type.startswith("barker"):
        if code_type == "barker":
            L = N
        else:
            L = int(code_type.replace("barker", ""))
        if L not in barker_table:
            raise ValueError(f"Barker supports lengths {sorted(barker_table.keys())}, got {L}")
        return np.array(barker_table[L], dtype=np.complex128)

    # ----------------------------
    # Frank / P1..P4 polyphase
    # ----------------------------
    if code_type in ("frank", "p1", "p2", "p3", "p4"):
        M = int(np.sqrt(N))
        if M * M != N:
            raise ValueError(f"{code_type.upper()} code requires N = M^2 (e.g., 16, 25, 36, 49, 64, 81...)")

        # indices 0..M-1
        p = np.arange(M)
        r = np.arange(M)
        P, R = np.meshgrid(p, r, indexing="ij")  # P rows, R cols

        if code_type == "frank":
            # Standard Frank: phi = 2π * P*R / M
            phi = 2 * np.pi * (P * R) / M

        elif code_type == "p1":
            # P1 family variant (commonly referenced polyphase radar code)
            phi = (np.pi / M) * (P * R) + (np.pi / M) * (R * (R + 1) / 2)

        elif code_type == "p2":
            phi = (np.pi / M) * (P * R) + (np.pi / M) * (P * (P + 1) / 2)

        elif code_type == "p3":
            phi = (np.pi / M) * (P * R) + (np.pi / M) * (R ** 2)

        elif code_type == "p4":
            phi = (np.pi / M) * (P * R) + (np.pi / (2 * M)) * (P ** 2 + R ** 2)

        return np.exp(1j * phi).reshape(-1).astype(np.complex128)

    raise ValueError("Unsupported code_type. Use: barker, frank, p1, p2, p3, p4")

# this function generates adc baseband signal by oversampling it.
# adc has Nchips * sps values (64 * 16), meaning phase changes over one pulse 64 times so adc data will have same phase for 16 samples.
# and adc has 1024 samples
def oversample_chips(chip_seq: np.ndarray, sps: int): # chip = [c0, c1, c2, c3 .... c64]
    return np.repeat(chip_seq, sps) # oversampled_chip = [c0, c0, c0, c0 16 times, .... c64]

# Matched-filter inputs are normalized so all waveforms have the same total energy, 
# ensuring that differences in output amplitude reflect waveform properties, not transmit power.
def normalize_energy(x: np.ndarray):
    e = np.sum(np.abs(x) ** 2) + 1e-12
    return x / np.sqrt(e)

# ----------------------------
# Channel / impairments
# ----------------------------

# this function adds zeros equivalent of delay meaning no pulse yet.
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

def apply_const_phase(x: np.ndarray, phi_deg: float):
    return x * np.exp(1j * np.deg2rad(phi_deg))

def apply_cfo(x, f_off_hz, fs):
    n = np.arange(len(x))
    return x * np.exp(1j * 2*np.pi * f_off_hz * n / fs)

def phase_deg_rounded(x: np.ndarray, decimals: int = 0):
    #Convert complex samples to phase in degrees and round.
    phase_deg = np.degrees(np.angle(x))
    return np.round(phase_deg, decimals=decimals)

# ----------------------------
# Matched filter + metrics
# ----------------------------
def matched_filter(rx: np.ndarray, ref: np.ndarray):
    # correlation = convolve(rx, conj(reverse(ref)))
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
# MAIN (simple and direct)
# ----------------------------
if __name__ == "__main__":

    # chip is the each phase index value on a pulse not each phase coded pulse
    # for N = 64 one pulse has 64 chips on it.
    # So chip is: This pulse is divided into 64 time segments, and each segment has a fixed phase.”
    
    # ---- Parameters you care about ----
    code_type       = "p4"
    Nchips          = 64
    sps             = 16 # pulse has 64 chip segments so phase will change 64 times so during 16 samples phase will be same. Overall adc will have 1024 samples with each 16 has same phase!!!
    fs              = 100.0e6 # Tpulse = Nchips * sps / fs, 64* 16 / 1mhz = 1ms pulse, 64*16/100mhz = 10microsec pulse
    delay_samp      = 400
    fd_hz           = 1000.0  
    snr_db          = 30.0
    lo_phase_shift  = 180.0
     
    # ---- Build reference waveform (the radar's known transmit) ----
    code = make_phase_code(code_type, Nchips) # 1x64 array
    
    # this oversampled signal has constant amplitude so plot is straight line for amplitude vs time.
    # the phase of the signal is changing!!!
    ref  = oversample_chips(code, sps) # adc baseband if generation # 1x1024 array
    ref  = normalize_energy(ref)
    ref_phase_deg = phase_deg_rounded(code) # Convert complex phase numbers to rad and degrees, round makes it integers value
    print("Coded tx pulse phase values: ", ref_phase_deg)
    
    
    # ---- Build drfm type of pulse with constant lo phase offset from original phase code
    drfm_code = apply_const_phase(code, lo_phase_shift)    
    drfm  = oversample_chips(drfm_code, sps) # adc baseband if generation # 1x1024 array
    drfm  = normalize_energy(drfm)
    drfm_phase_deg = phase_deg_rounded(drfm_code)
    print("DRFM pulse constant phase shifted values", drfm_phase_deg)


    # ---- Build drfm type of pulse with CFO (phase ramp) from original phase code
    # IMPORTANT: CFO must be applied on the oversampled waveform using fs.
    drfm2  = oversample_chips(code, sps) # adc baseband if generation # 1x1024 array
    drfm2  = normalize_energy(drfm2)
    drfm2  = apply_cfo(drfm2, 100e3, fs)   # CFO applied correctly on 1024-sample waveform
    drfm_code2 = drfm2                   # keep your variable name, but now it is the CFO-impaired waveform
    drfm2_phase_deg = phase_deg_rounded(drfm_code2)
    print("DRFM pulse phase shifted values", drfm2_phase_deg)
    
    
    # ---- Build "normal" uncoded pulse (same length) ----
    normal = oversample_chips(np.ones(Nchips, dtype=np.complex128), sps)
    normal = normalize_energy(normal)
    normal_phase_deg = phase_deg_rounded(np.angle(normal))
    print("Uncoded pulse phase values", normal_phase_deg)


    # ---- 3) Simulate received echoes ----
    # Case A: target echo matches the coded waveform
    rx_coded = apply_delay(ref, delay_samp)
    rx_coded = apply_doppler(rx_coded, fd_hz, fs) # shift freq through pulse
    rx_coded = add_awgn(rx_coded, snr_db)

    # Case B: target echo is an uncoded pulse (mismatch case)
    rx_normal = apply_delay(normal, delay_samp)
    rx_normal = apply_doppler(rx_normal, fd_hz, fs)
    rx_normal = add_awgn(rx_normal, snr_db)
    
    # Case C: target echo is drfm with phase shift
    rx_drfm = apply_delay(drfm, delay_samp)
    rx_drfm = apply_doppler(rx_drfm, fd_hz, fs)
    rx_drfm = add_awgn(rx_drfm, snr_db)

    # Case D: target echo is drfm with phase shift
    rx_drfm2 = apply_delay(drfm2, delay_samp)
    rx_drfm2 = apply_doppler(rx_drfm2, fd_hz, fs)
    rx_drfm2 = add_awgn(rx_drfm2, snr_db)

    # ---- 4) Matched filter both using the CODED reference ----
    y_coded  = matched_filter(rx_coded, ref)
    y_normal = matched_filter(rx_normal, ref)
    y_drfm   = matched_filter(rx_drfm, ref)
    y_drfm2  = matched_filter(rx_drfm2, ref)

    # ---- 5) Compute and print results ----
    peak_c, idx_c, pslr_c = peak_and_pslr_db(y_coded)
    peak_n, idx_n, pslr_n = peak_and_pslr_db(y_normal)
    peak_d, idx_d, pslr_d = peak_and_pslr_db(y_drfm)
    peak_e, idx_e, pslr_e = peak_and_pslr_db(y_drfm2)



    ratio_db = 20 * np.log10((peak_c + 1e-12) / (peak_n + 1e-12))
    percent  = 100.0 * (peak_n / (peak_c + 1e-12))

    print("\n=== MATCHED FILTER TEST (coded reference) ===")
    print(f"Code: {code_type}, Nchips={Nchips}, sps={sps}, SNR={snr_db} dB, fd={fd_hz} Hz, delay={delay_samp} samp")

    print("\nCase A: Received echo IS coded (matched)")
    print(f"  Peak magnitude : {peak_c:.6f}  (index {idx_c})")
    print(f"  PSLR           : {pslr_c:.2f} dB")

    print("\nCase B: Received echo is NORMAL pulse (mismatched)")
    print(f"  Peak magnitude : {peak_n:.6f}  (index {idx_n})")
    print(f"  PSLR           : {pslr_n:.2f} dB")

    print("\nCase C: Received echo is DRFM pulse with phase shift {lo_phase_shift:.2f} degrees")
    print(f"  Peak magnitude : {peak_d:.6f}  (index {idx_d})")
    print(f"  PSLR           : {pslr_d:.2f} dB")

    print("\nCase D: Received echo is DRFM with phase shift")
    print(f"  Peak magnitude : {peak_e:.6f}  (index {idx_e})")
    print(f"  PSLR           : {pslr_e:.2f} dB")

    print("\nDiscrimination (how much better the coded echo matches):")
    print(f"  Peak ratio     : {ratio_db:.2f} dB  (coded peak vs normal peak)")
    print(f"  Normal as %    : {percent:.1f}% of coded peak\n")



    # ---- after you compute y_coded, y_normal and peak_c/idx_c ... ----

    # Optional: normalize matched filter magnitudes for easier visual comparison
    yc_mag = np.abs(y_coded)
    yn_mag = np.abs(y_normal)

    # Optional: dB view (recommended)
    yc_db = 20*np.log10(yc_mag + 1e-12)
    yn_db = 20*np.log10(yn_mag + 1e-12)

    # Make a 4 rows x 2 columns figure
    fig, axs = plt.subplots(4, 2, figsize=(14, 10), constrained_layout=True)

    # =========================
    # LEFT COLUMN (your existing plots)
    # =========================

    # Row 0, Col 0: phase code (chips)
    x = np.arange(Nchips)
    axs[0, 0].stem(x, ref_phase_deg)
    axs[0, 0].set_xlabel("Chip index")
    axs[0, 0].set_ylabel("Phase (deg)")
    axs[0, 0].set_title("Polyphase Code (chips)")
    axs[0, 0].grid(True)

    # Row 1, Col 0: coded RF waveform (zoomed)
    fc = 6e9
    fs_rf = 50e9
    t = np.arange(len(ref)) / fs_rf
    rf_coded = np.real(ref * np.exp(1j * 2*np.pi*fc*t))
    axs[1, 0].plot(rf_coded[:2000])
    axs[1, 0].set_title("Coded Tx Pulse RF (zoomed)")
    axs[1, 0].set_xlabel("Sample index")
    axs[1, 0].set_ylabel("Amplitude")
    axs[1, 0].grid(True)

    # Row 2, Col 0: uncoded pulse phase (baseband) — this is constant 0 deg
    x2 = np.arange(len(normal_phase_deg))  # normal_phase_deg is length 1024
    axs[2, 0].plot(x2, normal_phase_deg)   # stem also ok, but plot is faster for 1024 points
    axs[2, 0].set_xlabel("Sample index")
    axs[2, 0].set_ylabel("Phase (deg)")
    axs[2, 0].set_title("Normal Pulse (Uncoded) Phase")
    axs[2, 0].grid(True)

    # Row 3, Col 0: uncoded RF waveform (zoomed)
    t2 = np.arange(len(normal)) / fs_rf
    rf_uncoded = np.real(normal * np.exp(1j * 2*np.pi*fc*t2))
    axs[3, 0].plot(rf_uncoded[:2000])
    axs[3, 0].set_title("Uncoded Tx Pulse RF (zoomed)")
    axs[3, 0].set_xlabel("Sample index")
    axs[3, 0].set_ylabel("Amplitude")
    axs[3, 0].grid(True)

    # =========================
    # RIGHT COLUMN (matched filter results)
    # =========================

    # Row 0, Col 1: matched filter output for coded echo (|y| in dB, normalized)
    axs[0, 1].plot(yc_db)
    axs[0, 1].axvline(idx_c, linestyle="--")  # peak marker
    axs[0, 1].set_title("Matched Filter Output (coded echo) |y| (dB, norm)")
    axs[0, 1].set_xlabel("Lag / output sample index")
    axs[0, 1].set_ylabel("Magnitude (dB)")
    axs[0, 1].grid(True)
    axs[0, 1].set_ylim(-60, 5)  # adjust as you like

    # Row 1, Col 1: matched filter output for uncoded echo (|y| in dB, normalized)
    axs[1, 1].plot(yn_db)
    axs[1, 1].axvline(idx_n, linestyle="--")  # peak marker
    axs[1, 1].set_title("Matched Filter Output (uncoded echo) |y| (dB, norm)")
    axs[1, 1].set_xlabel("Lag / output sample index")
    axs[1, 1].set_ylabel("Magnitude (dB)")
    axs[1, 1].grid(True)
    axs[1, 1].set_ylim(-60, 5)

    # Row 2, Col 1: overlay comparison (coded vs uncoded) in dB
    axs[2, 1].plot(yc_db, label="coded echo")
    axs[2, 1].plot(yn_db, label="uncoded echo")
    axs[2, 1].set_title("Matched Filter Comparison (dB, norm)")
    axs[2, 1].set_xlabel("Lag / output sample index")
    axs[2, 1].set_ylabel("Magnitude (dB)")
    axs[2, 1].grid(True)
    axs[2, 1].legend()
    axs[2, 1].set_ylim(-60, 5)

    # Row 3, Col 1: zoom around coded peak to see sidelobes better
    zoom = 200  # samples around peak
    lo = max(0, idx_c - zoom)
    hi = min(len(yc_db), idx_c + zoom)
    axs[3, 1].plot(np.arange(lo, hi), yc_db[lo:hi])
    axs[3, 1].axvline(idx_c, linestyle="--")
    axs[3, 1].set_title("Coded MF Output (zoom around peak)")
    axs[3, 1].set_xlabel("Lag / output sample index")
    axs[3, 1].set_ylabel("Magnitude (dB)")
    axs[3, 1].grid(True)
    axs[3, 1].set_ylim(-60, 5)

    plt.show()