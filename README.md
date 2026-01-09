# Polyphase Pulse Compression & DRFM Discrimination Simulation

This repository contains a **Python baseband radar simulation** demonstrating **phase-coded pulse compression**, **matched filtering**, and **discrimination against mismatched / DRFM-like echoes**.  
The implementation is intentionally radar-realistic and chip-accurate.

---

<img width="2878" height="1682" alt="Image" src="https://github.com/user-attachments/assets/9eab0037-0afb-46c9-8e89-2621109b2718" />

<img width="2780" height="1710" alt="Image" src="https://github.com/user-attachments/assets/ebc8d651-3857-4d33-a395-1322c371e27f" />

## Purpose

The script evaluates how a **matched filter** responds to different received pulse types when using a **phase-coded reference waveform**:

- Correctly coded target echo (matched)
- Uncoded rectangular pulse (mismatched)
- DRFM echo with constant phase rotation
- DRFM echo with carrier frequency offset (CFO)

This directly illustrates **processing gain, mismatch loss, and PSLR behavior** relevant to pulse radar and ECCM analysis.

---

## Signal Model

- One radar pulse is divided into **Nchips phase segments**
- Each chip has **constant phase**
- ADC oversampling is modeled using `sps` (samples per chip)
- Total ADC samples per pulse = `Nchips × sps`
- All waveforms are **energy-normalized** for fair comparison

---

## Supported Phase Codes

- Barker (BPSK): 2, 3, 4, 5, 7, 11, 13
- Frank code
- Polyphase codes: P1, P2, P3, P4  
  *(Require `Nchips = M²`, e.g. 16, 25, 36, 64)*

---

## Channel Impairments

- Time delay (range)
- Doppler shift
- Carrier frequency offset (CFO)
- Constant phase offset
- Additive white Gaussian noise (AWGN)

---

## Processing Chain

1. Phase code generation  
2. Chip oversampling (ADC model)  
3. Impairment injection  
4. Matched filtering (time-domain correlation)  
5. Metric extraction:
   - Peak magnitude
   - Peak Sidelobe Level Ratio (PSLR)
   - Coded vs uncoded peak ratio (dB)

---

## Outputs & Plots

- Chip-level phase structure
- RF waveform reconstruction (illustrative)
- Matched filter magnitude (linear & dB)
- Coded vs uncoded comparison
- Sidelobe zoom around main peak

---

## Key Insight

Correct phase coding produces:
- Strong compression peak
- High processing gain
- Low sidelobes

Uncoded or DRFM-distorted echoes exhibit:
- Reduced matched-filter peak
- Increased sidelobes
- Measurable discrimination loss

---

## Dependencies

- Python 3.x  
- NumPy  
- Matplotlib

---

## Intended Use

- Pulse radar waveform analysis
- Phase-coded compression studies
- DRFM / ECCM experimentation
- Educational and research purposes
