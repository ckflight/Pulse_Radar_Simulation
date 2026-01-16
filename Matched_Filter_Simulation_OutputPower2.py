import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons, CheckButtons

# ============================================================
# Constants / helpers
# ============================================================
C0 = 299_792_458.0
ERP_TO_EIRP_DB = 2.15  # EIRP = ERP + 2.15 dB (dipole vs isotropic)

def dbm_to_w(dbm: float) -> float:
    return 1e-3 * 10 ** (dbm / 10.0)

def scale_to_power_dbm(x: np.ndarray, p_dbm: float, R_ohm: float = 50.0) -> np.ndarray:
    """Scale complex samples so mean(|x|^2)=Vrms^2 corresponds to target power into R."""
    P_w = dbm_to_w(p_dbm)
    target_vrms2 = P_w * R_ohm
    cur_vrms2 = np.mean(np.abs(x) ** 2) + 1e-300
    return x * np.sqrt(target_vrms2 / cur_vrms2)

def complex_white_noise(rng: np.random.Generator, n: int) -> np.ndarray:
    """Unit-variance complex Gaussian: E|x|^2 = 1."""
    return (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2)

def fspl_db(f_hz: float, d_km: float) -> float:
    """Free-space path loss (MHz, km form)."""
    f_mhz = f_hz / 1e6
    return 32.45 + 20*np.log10(f_mhz + 1e-300) + 20*np.log10(d_km + 1e-300)

def thermal_noise_power_dbm(bw_hz: float, nf_db: float) -> float:
    """N(dBm) = -174 + 10log10(B) + NF  (290 K)"""
    return -174.0 + 10*np.log10(bw_hz + 1e-300) + nf_db

def radar_echo_power_dbm_monostatic_eirp(
    radar_eirp_dbm: float,
    radar_rx_gain_dbi: float,
    target_rcs_m2: float,
    f_hz: float,
    range_km: float,
    losses_db: float = 0.0
) -> float:
    """
    Monostatic radar equation using EIRP = Pt + Gt:
    Pr = EIRP + Gr + 20log10(lambda) + 10log10(sigma) - 30log10(4pi) - 40log10(R) - L
    """
    lam = C0 / f_hz
    R_m = range_km * 1000.0
    term = 20*np.log10(lam + 1e-300) + 10*np.log10(target_rcs_m2 + 1e-300)
    const = 30*np.log10(4*np.pi)
    rterm = 40*np.log10(R_m + 1e-300)
    return radar_eirp_dbm + radar_rx_gain_dbi + term - const - rterm - losses_db

def jammer_received_power_dbm_eirp(
    jammer_eirp_dbm: float,
    radar_gain_toward_jammer_dbi: float,
    f_hz: float,
    range_km: float,
    losses_db: float = 0.0
) -> float:
    """One-way jammer -> radar: Pr = EIRP_jammer + G_radar(toward jammer) - FSPL - L"""
    return jammer_eirp_dbm + radar_gain_toward_jammer_dbi - fspl_db(f_hz, range_km) - losses_db

def matched_filter(rx: np.ndarray, ref: np.ndarray) -> np.ndarray:
    h = ref[::-1].conj()
    return np.convolve(rx, h, mode="full")

def peak_floor_metrics(y: np.ndarray, guard: int = 30):
    mag = np.abs(y)
    pk = int(np.argmax(mag))
    peak = mag[pk] + 1e-300

    mask = np.ones_like(mag, dtype=bool)
    lo = max(0, pk - guard)
    hi = min(len(mag), pk + guard + 1)
    mask[lo:hi] = False

    floor = np.median(mag[mask]) + 1e-300
    p2f_db = 20*np.log10(peak / floor)
    return pk, 20*np.log10(peak), 20*np.log10(floor), p2f_db

# ============================================================
# Simple waveform: Barker-13 (fast & realistic enough for demo)
# ============================================================
def barker13():
    seq = [1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1]
    return np.array(seq, dtype=np.complex128)

def oversample(seq: np.ndarray, sps: int) -> np.ndarray:
    return np.repeat(seq, sps)

def apply_fast_time_doppler(x: np.ndarray, fd_hz: float, fs: float):
    """Fast-time Doppler within a pulse."""
    if fd_hz == 0.0:
        return x
    n = np.arange(len(x))
    return x * np.exp(1j * 2*np.pi*fd_hz*n/fs)

# ============================================================
# Core simulation: N pulses + integration
# ============================================================
def run_sim_np(
    radar_erp_dbm: float,
    jammer_erp_dbm: float,
    range_km: float,
    rcs_m2: float,
    g_rx_dbi: float,
    g_toward_jammer_dbi: float,
    f_hz: float,
    fs: float,
    b_rx_hz: float,
    nf_db: float,
    fd_hz: float,
    prf_hz: float,
    delay_samp: int,
    Np: int,
    integ_mode: str,                 # "coherent" | "noncoherent_mag" | "noncoherent_power"
    jammer_coherent: bool,           # if True: same jammer waveform each pulse (repeater-like feel)
    seed: int = 0,
    losses_db: float = 0.0
):
    # ERP->EIRP
    radar_eirp_dbm  = radar_erp_dbm  + ERP_TO_EIRP_DB
    jammer_eirp_dbm = jammer_erp_dbm + ERP_TO_EIRP_DB

    # Receiver input powers
    pr_echo_dbm = radar_echo_power_dbm_monostatic_eirp(
        radar_eirp_dbm=radar_eirp_dbm,
        radar_rx_gain_dbi=g_rx_dbi,
        target_rcs_m2=rcs_m2,
        f_hz=f_hz,
        range_km=range_km,
        losses_db=losses_db
    )
    pr_jam_dbm = jammer_received_power_dbm_eirp(
        jammer_eirp_dbm=jammer_eirp_dbm,
        radar_gain_toward_jammer_dbi=g_toward_jammer_dbi,
        f_hz=f_hz,
        range_km=range_km,
        losses_db=losses_db
    )
    pr_th_dbm = thermal_noise_power_dbm(b_rx_hz, nf_db)

    # Reference (fixed)
    ref = oversample(barker13(), sps=8)
    ref = ref / (np.abs(ref).max() + 1e-300)
    Nref = len(ref)

    # Echo template (fast-time Doppler) scaled to received echo power
    echo_fast = apply_fast_time_doppler(ref.copy(), fd_hz, fs)
    echo_fast = scale_to_power_dbm(echo_fast, pr_echo_dbm)

    # Buffer sizing
    rx_len = delay_samp + Nref + 400
    y_len = rx_len + Nref - 1  # "full" convolution length

    # PRI and pulse-to-pulse phase rotation
    prf_hz = max(prf_hz, 1e-6)
    Tpri = 1.0 / prf_hz
    dphi = 2*np.pi*fd_hz*Tpri  # phase advance per pulse (very important for coherent integration realism)

    # RNG (deterministic per slider position)
    rng = np.random.default_rng(seed)

    # Jammer waveform across pulses
    jammer_once = None
    if jammer_coherent:
        jammer_once = scale_to_power_dbm(complex_white_noise(rng, rx_len), pr_jam_dbm)

    # Integration accumulator
    if integ_mode == "coherent":
        y_int = np.zeros(y_len, dtype=np.complex128)
    else:
        y_int = np.zeros(y_len, dtype=np.float64)

    # Also keep one representative rx (for the top plot) – pulse 0
    rx0 = None

    for p in range(Np):
        rx = np.zeros(rx_len, dtype=np.complex128)

        # Echo pulse-to-pulse phase rotation (constant across the pulse)
        echo_p = echo_fast * np.exp(1j * dphi * p)

        rx[delay_samp:delay_samp+Nref] += echo_p

        # Jammer
        if jammer_coherent:
            jammer = jammer_once
        else:
            jammer = scale_to_power_dbm(complex_white_noise(rng, rx_len), pr_jam_dbm)

        # Thermal noise (always uncorrelated pulse-to-pulse)
        th = scale_to_power_dbm(complex_white_noise(rng, rx_len), pr_th_dbm)

        rx = rx + jammer + th

        if p == 0:
            rx0 = rx.copy()

        y = matched_filter(rx, ref)

        if integ_mode == "coherent":
            y_int += y
        elif integ_mode == "noncoherent_mag":
            y_int += np.abs(y)
        elif integ_mode == "noncoherent_power":
            y_int += (np.abs(y) ** 2)
        else:
            raise ValueError("Unknown integ_mode")

    # Convert noncoherent power to magnitude-like for plotting (optional but intuitive):
    if integ_mode == "noncoherent_power":
        y_plot = np.sqrt(y_int + 1e-300)  # so 20log10(.) still looks like amplitude scale
    elif integ_mode == "noncoherent_mag":
        y_plot = y_int
    else:
        y_plot = y_int

    pk, peak_db, floor_db, p2f_db = peak_floor_metrics(y_plot, guard=30)

    # Ratios (input domain)
    J_over_S = pr_jam_dbm - pr_echo_dbm
    N_over_S = pr_th_dbm  - pr_echo_dbm
    J_over_N = pr_jam_dbm - pr_th_dbm

    # Expected peak index (for a delayed insert + matched filter)
    expected_pk = delay_samp + (Nref - 1)

    # Ideal coherent processing gain for white noise (rule of thumb)
    coh_gain_db = 10.0*np.log10(max(Np, 1))

    return {
        "rx0": rx0,
        "y_plot": y_plot,
        "pk": pk,
        "expected_pk": expected_pk,
        "floor_db": floor_db,
        "pr_echo_dbm": pr_echo_dbm,
        "pr_jam_dbm": pr_jam_dbm,
        "pr_th_dbm": pr_th_dbm,
        "peak_db": peak_db,
        "p2f_db": p2f_db,
        "J_over_S": J_over_S,
        "N_over_S": N_over_S,
        "J_over_N": J_over_N,
        "ref_len": Nref,
        "dphi_deg": (dphi * 180.0 / np.pi) % 360.0,
        "coh_gain_db": coh_gain_db,
        "rx_len": rx_len,
        "y_len": len(y_plot),
    }

# ============================================================
# UI / main
# ============================================================
def main():
    # Fixed constants
    f_hz = 6e9
    fs = 100e6
    b_rx_hz = 100e6
    nf_db = 6.0
    delay_samp = 220
    g_rx_dbi = 35.0
    base_seed = 0

    # Initial values
    radar_erp0 = 100.0
    jammer_erp0 = 60.0
    range0 = 50.0
    rcs0 = 3.0
    g_toward_jammer0 = 0.0
    fd0 = 1000.0
    prf0 = 1000.0
    Np0 = 32

    # UI style
    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
    })

    fig = plt.figure(figsize=(13, 9), dpi=110)
    # more bottom space for sliders + radio/check boxes
    fig.subplots_adjust(left=0.06, right=0.985, top=0.94, bottom=0.40, hspace=0.35)

    gs = fig.add_gridspec(nrows=3, ncols=1, height_ratios=[2.1, 2.2, 1.2])
    ax_rx = fig.add_subplot(gs[0, 0])
    ax_y  = fig.add_subplot(gs[1, 0])
    ax_info = fig.add_subplot(gs[2, 0])
    ax_info.axis("off")

    # ---- Slider layout
    slider_left  = 0.18
    slider_width = 0.74
    slider_h     = 0.028
    slider_gap   = 0.010
    slider_bottom0 = 0.05

    # 7 sliders
    ax_s_gjam   = fig.add_axes([slider_left, slider_bottom0 + (slider_h + slider_gap)*6, slider_width, slider_h])
    ax_s_radar  = fig.add_axes([slider_left, slider_bottom0 + (slider_h + slider_gap)*5, slider_width, slider_h])
    ax_s_jammer = fig.add_axes([slider_left, slider_bottom0 + (slider_h + slider_gap)*4, slider_width, slider_h])
    ax_s_range  = fig.add_axes([slider_left, slider_bottom0 + (slider_h + slider_gap)*3, slider_width, slider_h])
    ax_s_rcs    = fig.add_axes([slider_left, slider_bottom0 + (slider_h + slider_gap)*2, slider_width, slider_h])
    ax_s_fd     = fig.add_axes([slider_left, slider_bottom0 + (slider_h + slider_gap)*1, slider_width, slider_h])
    ax_s_prf    = fig.add_axes([slider_left, slider_bottom0 + (slider_h + slider_gap)*0, slider_width, slider_h])

    s_gjam   = Slider(ax_s_gjam,   "Radar gain toward jammer (dBi)", -40.0, 40.0, valinit=g_toward_jammer0, valstep=0.5)
    s_radar  = Slider(ax_s_radar,  "Radar ERP (dBm)",  40.0, 140.0, valinit=radar_erp0, valstep=0.5)
    s_jammer = Slider(ax_s_jammer, "Jammer ERP (dBm)",  0.0, 120.0, valinit=jammer_erp0, valstep=0.5)
    s_range  = Slider(ax_s_range,  "Range (km)",        1.0, 150.0, valinit=range0, valstep=0.5)
    s_rcs    = Slider(ax_s_rcs,    "RCS (m^2)",         0.1, 50.0,  valinit=rcs0)
    s_fd     = Slider(ax_s_fd,     "Doppler fd (Hz)",   -30e3, 30e3, valinit=fd0, valstep=10.0)
    s_prf    = Slider(ax_s_prf,    "PRF (Hz)",          50.0, 50e3,  valinit=prf0, valstep=10.0)

    for s in (s_gjam, s_radar, s_jammer, s_range, s_rcs, s_fd, s_prf):
        s.label.set_fontsize(10)
        s.valtext.set_fontsize(10)

    # ---- Np slider as an integer (separate small bar, right side)
    ax_s_np = fig.add_axes([0.80, 0.33, 0.17, 0.03])
    s_np = Slider(ax_s_np, "Np", 1, 256, valinit=Np0, valstep=1)
    s_np.label.set_fontsize(10)
    s_np.valtext.set_fontsize(10)

    # ---- Radio buttons (integration mode)
    ax_radio = fig.add_axes([0.06, 0.30, 0.20, 0.10])
    radio = RadioButtons(
        ax_radio,
        labels=("coherent", "noncoherent_mag", "noncoherent_power"),
        active=0
    )
    for t in radio.labels:
        t.set_fontsize(9)
    ax_radio.set_title("Integration mode", fontsize=10, pad=6)

    # ---- Check button (jammer coherence)
    ax_check = fig.add_axes([0.06, 0.24, 0.20, 0.05])
    check = CheckButtons(ax_check, labels=("Jammer coherent across pulses",), actives=(False,))
    for t in check.labels:
        t.set_fontsize(9)

    # ---- Run first sim
    def get_modes():
        integ_mode = radio.value_selected
        jammer_coh = check.get_status()[0]
        return integ_mode, jammer_coh

    out = run_sim_np(
        radar_erp_dbm=radar_erp0,
        jammer_erp_dbm=jammer_erp0,
        range_km=range0,
        rcs_m2=rcs0,
        g_rx_dbi=g_rx_dbi,
        g_toward_jammer_dbi=g_toward_jammer0,
        f_hz=f_hz,
        fs=fs,
        b_rx_hz=b_rx_hz,
        nf_db=nf_db,
        fd_hz=fd0,
        prf_hz=prf0,
        delay_samp=delay_samp,
        Np=Np0,
        integ_mode="coherent",
        jammer_coherent=False,
        seed=base_seed
    )

    # ---- Plot init: RX (pulse 0)
    rx_db = 20*np.log10(np.abs(out["rx0"]) + 1e-300)
    (l_rx,) = ax_rx.plot(rx_db, linewidth=1.1)
    ax_rx.set_title("Receiver input (pulse 0): 20log10(|rx[n]|)  = echo + jammer + thermal")
    ax_rx.set_xlabel("Sample index")
    ax_rx.set_ylabel("dB (relative)")
    ax_rx.grid(True, alpha=0.35)

    # ---- Plot init: MF output (integrated)
    y_db = 20*np.log10(np.abs(out["y_plot"]) + 1e-300)
    (l_y,) = ax_y.plot(y_db, linewidth=1.1)
    ax_y.set_title("Matched filter output (integrated): 20log10(|y[k]|)")
    ax_y.set_xlabel("Lag index")
    ax_y.set_ylabel("dB (relative)")
    ax_y.grid(True, alpha=0.35)

    # Found peak markers
    vline_pk = ax_y.axvline(out["pk"], linestyle="--", linewidth=1.0)
    pmark = ax_y.plot(out["pk"], y_db[out["pk"]], marker="o", markersize=5)[0]

    # Expected peak index marker (from delay & ref_len)
    exp_line = ax_y.axvline(out["expected_pk"], color="orange", linestyle="-", linewidth=2.0, alpha=0.9)
    exp_text = ax_y.text(
        out["expected_pk"], np.max(y_db) - 2,
        "expected",
        color="orange",
        rotation=90,
        va="top",
        ha="right",
        fontsize=9,
        bbox=dict(facecolor="white", alpha=0.65, edgecolor="none", pad=1.2)
    )

    # Noise floor line (horizontal)
    floor_line = ax_y.axhline(out["floor_db"], color="orange", linestyle="--", linewidth=2.0, alpha=0.9)
    floor_text = ax_y.text(
        0.99, 0.02, f"floor = {out['floor_db']:.1f} dB",
        transform=ax_y.transAxes,
        ha="right", va="bottom",
        fontsize=9,
        bbox=dict(facecolor="white", alpha=0.65, edgecolor="none", pad=1.2)
    )

    # Info box
    info_text = ax_info.text(
        0.01, 0.98, "",
        va="top", ha="left",
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.95, edgecolor="#cccccc")
    )

    def refresh_info(o, integ_mode, jammer_coh, Np, prf_hz, fd_hz):
        info_text.set_text(
            "\n".join([
                f"f={f_hz/1e9:.2f} GHz   fs={fs/1e6:.1f} MSps   B_RX={b_rx_hz/1e6:.1f} MHz   NF={nf_db:.1f} dB",
                f"delay_samp={delay_samp}   ref_len={o['ref_len']}   expected_pk={o['expected_pk']}",
                f"Np={Np}   PRF={prf_hz:.1f} Hz   fd={fd_hz:.1f} Hz   dphi/pulse={o['dphi_deg']:.1f} deg",
                f"Integration={integ_mode}   Jammer_coherent={jammer_coh}",
                "",
                f"S_in (echo)   : {o['pr_echo_dbm']:.2f} dBm",
                f"J_in (jammer) : {o['pr_jam_dbm']:.2f} dBm",
                f"N_in (thermal): {o['pr_th_dbm']:.2f} dBm",
                "",
                f"J/S_in={o['J_over_S']:.2f} dB   N/S_in={o['N_over_S']:.2f} dB   J/N={o['J_over_N']:.2f} dB",
                f"MF peak={o['peak_db']:.2f} dB   MF floor={o['floor_db']:.2f} dB   Peak-to-floor={o['p2f_db']:.2f} dB",
                f"Ideal coherent gain (white noise) ~ +{o['coh_gain_db']:.2f} dB"
            ])
        )

    integ_mode0, jammer_coh0 = get_modes()
    refresh_info(out, integ_mode0, jammer_coh0, int(s_np.val), float(s_prf.val), float(s_fd.val))

    # ---- Update function
    def update(_=None):
        integ_mode, jammer_coh = get_modes()
        Np = int(s_np.val)

        o = run_sim_np(
            radar_erp_dbm=float(s_radar.val),
            jammer_erp_dbm=float(s_jammer.val),
            range_km=float(s_range.val),
            rcs_m2=float(s_rcs.val),
            g_rx_dbi=g_rx_dbi,
            g_toward_jammer_dbi=float(s_gjam.val),
            f_hz=f_hz,
            fs=fs,
            b_rx_hz=b_rx_hz,
            nf_db=nf_db,
            fd_hz=float(s_fd.val),
            prf_hz=float(s_prf.val),
            delay_samp=delay_samp,
            Np=Np,
            integ_mode=integ_mode,
            jammer_coherent=jammer_coh,
            seed=base_seed
        )

        # RX plot (pulse 0)
        rx_db2 = 20*np.log10(np.abs(o["rx0"]) + 1e-300)
        l_rx.set_data(np.arange(len(rx_db2)), rx_db2)
        ax_rx.set_xlim(0, len(rx_db2)-1)
        ax_rx.set_ylim(rx_db2.min()-3, rx_db2.max()+3)

        # MF plot (integrated)
        y_db2 = 20*np.log10(np.abs(o["y_plot"]) + 1e-300)
        l_y.set_data(np.arange(len(y_db2)), y_db2)
        ax_y.set_xlim(0, len(y_db2)-1)
        ax_y.set_ylim(y_db2.min()-3, y_db2.max()+3)

        # Found peak
        vline_pk.set_xdata([o["pk"], o["pk"]])
        pmark.set_data([o["pk"]], [y_db2[o["pk"]]])

        # Expected peak
        exp_line.set_xdata([o["expected_pk"], o["expected_pk"]])
        y_top = ax_y.get_ylim()[1]
        exp_text.set_position((o["expected_pk"], y_top - 2))

        # Noise floor (horizontal)
        floor_line.set_ydata([o["floor_db"], o["floor_db"]])
        floor_text.set_text(f"floor = {o['floor_db']:.1f} dB")

        refresh_info(o, integ_mode, jammer_coh, Np, float(s_prf.val), float(s_fd.val))
        fig.canvas.draw_idle()

    # Hook events
    for s in (s_gjam, s_radar, s_jammer, s_range, s_rcs, s_fd, s_prf, s_np):
        s.on_changed(update)
    radio.on_clicked(lambda _: update())
    check.on_clicked(lambda _: update())

    plt.show()

if __name__ == "__main__":
    main()
