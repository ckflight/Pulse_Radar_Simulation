import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# ============================================================
# Constants / helpers
# ============================================================
C0 = 299_792_458.0
ERP_TO_EIRP_DB = 2.15

def dbm_to_w(dbm: float) -> float:
    return 1e-3 * 10 ** (dbm / 10.0)

def scale_to_power_dbm(x: np.ndarray, p_dbm: float, R_ohm: float = 50.0) -> np.ndarray:
    """Scale complex samples so mean(|x|^2) corresponds to Vrms^2 for target power."""
    P_w = dbm_to_w(p_dbm)
    target_vrms2 = P_w * R_ohm
    cur_vrms2 = np.mean(np.abs(x) ** 2) + 1e-300
    return x * np.sqrt(target_vrms2 / cur_vrms2)

def complex_white_noise(n: int) -> np.ndarray:
    return (np.random.randn(n) + 1j * np.random.randn(n)) / np.sqrt(2)

def fspl_db(f_hz: float, d_km: float) -> float:
    """Free-space path loss (MHz, km form)."""
    f_mhz = f_hz / 1e6
    return 32.45 + 20*np.log10(f_mhz + 1e-300) + 20*np.log10(d_km + 1e-300)

def thermal_noise_power_dbm(bw_hz: float, nf_db: float) -> float:
    """N(dBm) = -174 + 10log10(B) + NF"""
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
    """One-way jammer -> radar: Pr = EIRP_jammer + G_radar - FSPL - L"""
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

def apply_doppler(x: np.ndarray, fd_hz: float, fs: float):
    if fd_hz == 0.0:
        return x
    n = np.arange(len(x))
    return x * np.exp(1j * 2*np.pi*fd_hz*n/fs)

# ============================================================
# Core simulation
# ============================================================
def run_sim(
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
    delay_samp: int,
    losses_db: float = 0.0
):
    # ERP->EIRP (education)
    radar_eirp_dbm  = radar_erp_dbm  + ERP_TO_EIRP_DB
    jammer_eirp_dbm = jammer_erp_dbm + ERP_TO_EIRP_DB

    # Input powers (receiver input)
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

    # Reference
    ref = oversample(barker13(), sps=8)  # fast
    ref = ref / (np.abs(ref).max() + 1e-300)

    # Echo (matched) + Doppler
    echo = apply_doppler(ref.copy(), fd_hz, fs)
    echo = scale_to_power_dbm(echo, pr_echo_dbm)

    # RX buffer
    rx_len = delay_samp + len(ref) + 400
    rx = np.zeros(rx_len, dtype=np.complex128)

    # Insert echo
    rx[delay_samp:delay_samp+len(echo)] += echo

    # Add jammer + thermal
    jammer = scale_to_power_dbm(complex_white_noise(rx_len), pr_jam_dbm)
    th     = scale_to_power_dbm(complex_white_noise(rx_len), pr_th_dbm)
    rx = rx + jammer + th

    # Matched filter
    y = matched_filter(rx, ref)
    pk, peak_db, floor_db, p2f_db = peak_floor_metrics(y, guard=30)

    # Ratios (input domain)
    J_over_S = pr_jam_dbm - pr_echo_dbm
    N_over_S = pr_th_dbm  - pr_echo_dbm
    J_over_N = pr_jam_dbm - pr_th_dbm

    return {
        "rx": rx, "y": y, "pk": pk,
        "pr_echo_dbm": pr_echo_dbm, "pr_jam_dbm": pr_jam_dbm, "pr_th_dbm": pr_th_dbm,
        "peak_db": peak_db, "floor_db": floor_db, "p2f_db": p2f_db,
        "J_over_S": J_over_S, "N_over_S": N_over_S, "J_over_N": J_over_N,
        "ref_len": len(ref)
    }

# ============================================================
# UI / main
# ============================================================
def main():
    np.random.seed(0)

    # Fixed constants
    f_hz = 6e9
    fs = 100e6
    b_rx_hz = 100e6
    nf_db = 6.0
    fd_hz = 1000.0
    delay_samp = 220
    g_rx_dbi = 35.0

    # Initial slider values
    radar_erp0 = 100.0
    jammer_erp0 = 60.0
    range0 = 50.0
    rcs0 = 3.0
    g_toward_jammer0 = 0.0

    # Figure styling
    plt.rcParams.update({
        "font.size": 8,
        "axes.titlesize": 10,
        "axes.labelsize": 8,
    })

    fig = plt.figure(figsize=(12, 8), dpi=100)

    # more bottom space so slider labels don't get clipped
    fig.subplots_adjust(left=0.06, right=0.985, top=0.95, bottom=0.36, hspace=0.35)

    gs = fig.add_gridspec(nrows=3, ncols=1, height_ratios=[2.2, 2.2, 1.2])

    ax_rx = fig.add_subplot(gs[0, 0])
    ax_y  = fig.add_subplot(gs[1, 0])
    ax_info = fig.add_subplot(gs[2, 0])
    ax_info.axis("off")

    # Slider area (clean block)
    slider_left = 0.14
    slider_width = 0.80
    slider_h = 0.028
    slider_gap = 0.012
    slider_bottom0 = 0.04

    ax_s_gjam   = fig.add_axes([slider_left, slider_bottom0 + (slider_h + slider_gap)*4, slider_width, slider_h])
    ax_s_radar  = fig.add_axes([slider_left, slider_bottom0 + (slider_h + slider_gap)*3, slider_width, slider_h])
    ax_s_jammer = fig.add_axes([slider_left, slider_bottom0 + (slider_h + slider_gap)*2, slider_width, slider_h])
    ax_s_range  = fig.add_axes([slider_left, slider_bottom0 + (slider_h + slider_gap)*1, slider_width, slider_h])
    ax_s_rcs    = fig.add_axes([slider_left, slider_bottom0 + (slider_h + slider_gap)*0, slider_width, slider_h])

    s_gjam   = Slider(ax_s_gjam,   "Radar gain toward jammer (dBi)", -40.0, 40.0, valinit=g_toward_jammer0, valstep=0.5)
    s_radar  = Slider(ax_s_radar,  "Radar ERP (dBm)",  40.0, 140.0, valinit=radar_erp0, valstep=0.5)
    s_jammer = Slider(ax_s_jammer, "Jammer ERP (dBm)",  0.0, 120.0, valinit=jammer_erp0, valstep=0.5)
    s_range  = Slider(ax_s_range,  "Range (km)",        1.0, 150.0, valinit=range0, valstep=0.5)
    s_rcs    = Slider(ax_s_rcs,    "RCS (m^2)",         0.1, 50.0,  valinit=rcs0)

    for s in (s_gjam, s_radar, s_jammer, s_range, s_rcs):
        s.label.set_fontsize(10)
        s.valtext.set_fontsize(10)

    # Initial run
    out = run_sim(
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
        fd_hz=fd_hz,
        delay_samp=delay_samp
    )

    # Plot init
    rx_db = 20*np.log10(np.abs(out["rx"]) + 1e-300)
    y_db  = 20*np.log10(np.abs(out["y"])  + 1e-300)

    (l_rx,) = ax_rx.plot(rx_db, linewidth=1.2)
    ax_rx.set_title("Receiver input |rx[n]| (dB, relative): echo + jammer + thermal")
    ax_rx.set_xlabel("Sample index")
    ax_rx.set_ylabel("dB")
    ax_rx.grid(True, alpha=0.35)

    (l_y,) = ax_y.plot(y_db, linewidth=1.2)

    # Found peak markers
    vline = ax_y.axvline(out["pk"], linestyle="--", linewidth=1.0)
    pmark = ax_y.plot(out["pk"], y_db[out["pk"]], marker="o", markersize=5)[0]

    # Expected peak line (from delay_samp and ref length)
    expected_pk = delay_samp + (out["ref_len"] - 1)
    exp_line = ax_y.axvline(expected_pk, color="orange", linestyle="-", linewidth=2.0, alpha=0.9)
    exp_text = ax_y.text(
        expected_pk, np.max(y_db) - 2,
        "expected",
        color="orange",
        rotation=90,
        va="top",
        ha="right",
        fontsize=9,
        bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=1.5)
    )

    # ============================================================
    # NEW: Horizontal floor line on matched-filter output
    # ============================================================
    floor_line = ax_y.axhline(out["floor_db"], color="orange", linestyle="-", linewidth=2.0, alpha=0.85)
    floor_text = ax_y.text(
        0.99, out["floor_db"],
        " floor",
        color="orange",
        va="bottom",
        ha="right",
        fontsize=9,
        transform=ax_y.get_yaxis_transform(),  # x in axes-fraction, y in data
        bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=1.2)
    )

    ax_y.set_title("Matched filter output |y[k]| (dB, relative)")
    ax_y.set_xlabel("Lag index")
    ax_y.set_ylabel("dB")
    ax_y.grid(True, alpha=0.35)

    # Info box
    info_text = ax_info.text(
        0.01, 0.98, "",
        va="top", ha="left",
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.92, edgecolor="#cccccc")
    )

    def refresh_text(o):
        info_text.set_text(
            "\n".join([
                f"f={f_hz/1e9:.2f} GHz   fs={fs/1e6:.1f} MSps   B_RX={b_rx_hz/1e6:.1f} MHz   NF={nf_db:.1f} dB",
                f"S_in (echo)   : {o['pr_echo_dbm']:.2f} dBm",
                f"J_in (jammer) : {o['pr_jam_dbm']:.2f} dBm",
                f"N_in (thermal): {o['pr_th_dbm']:.2f} dBm",
                "",
                f"J/S_in={o['J_over_S']:.2f} dB   N/S_in={o['N_over_S']:.2f} dB   J/N={o['J_over_N']:.2f} dB",
                f"MF peak={o['peak_db']:.2f} dB   MF floor={o['floor_db']:.2f} dB   Peak-to-floor={o['p2f_db']:.2f} dB",
                f"Ref length N={o['ref_len']} samples",
                f"Expected peak index = delay + (Nref-1) = {delay_samp} + ({o['ref_len']}-1) = {delay_samp + (o['ref_len']-1)}"
            ])
        )

    refresh_text(out)

    def update(_):
        # stable behavior while sliding
        np.random.seed(0)

        o = run_sim(
            radar_erp_dbm=s_radar.val,
            jammer_erp_dbm=s_jammer.val,
            range_km=s_range.val,
            rcs_m2=s_rcs.val,
            g_rx_dbi=g_rx_dbi,
            g_toward_jammer_dbi=s_gjam.val,
            f_hz=f_hz,
            fs=fs,
            b_rx_hz=b_rx_hz,
            nf_db=nf_db,
            fd_hz=fd_hz,
            delay_samp=delay_samp
        )

        rx_db2 = 20*np.log10(np.abs(o["rx"]) + 1e-300)
        y_db2  = 20*np.log10(np.abs(o["y"])  + 1e-300)

        l_rx.set_data(np.arange(len(rx_db2)), rx_db2)
        ax_rx.set_xlim(0, len(rx_db2)-1)
        ax_rx.set_ylim(rx_db2.min()-3, rx_db2.max()+3)

        l_y.set_data(np.arange(len(y_db2)), y_db2)
        ax_y.set_xlim(0, len(y_db2)-1)
        ax_y.set_ylim(y_db2.min()-3, y_db2.max()+3)

        # Found peak markers
        vline.set_xdata([o["pk"], o["pk"]])
        pmark.set_data([o["pk"]], [y_db2[o["pk"]]])

        # Expected peak markers
        expected_pk2 = delay_samp + (o["ref_len"] - 1)
        exp_line.set_xdata([expected_pk2, expected_pk2])

        # Keep expected label visible near top after ylim changes
        y_top = ax_y.get_ylim()[1]
        exp_text.set_position((expected_pk2, y_top - 2))

        # ============================================================
        # UPDATE: Horizontal floor line + label
        # ============================================================
        floor_line.set_ydata([o["floor_db"], o["floor_db"]])
        floor_text.set_position((0.99, o["floor_db"]))  # x is axes fraction due to transform

        refresh_text(o)
        fig.canvas.draw_idle()

    s_gjam.on_changed(update)
    s_radar.on_changed(update)
    s_jammer.on_changed(update)
    s_range.on_changed(update)
    s_rcs.on_changed(update)

    plt.show()

if __name__ == "__main__":
    main()
