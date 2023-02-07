import processes.foundation as fd
import processes.fitModel as fM
import processes.histogramAction as hA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lib.definition as proc

def main():
    data = fd.get_t1_data(9569, "Det1724")
    wave = data[0]["waveform"]["values"].nda[0]
    energy = 5
    pulse, start = proc.get_pulse(energy, 60, len(wave))
    wave = np.int64(wave) - np.int64(wave[1])
    wave = wave+pulse
    trap_energy, wf_trap = proc.apply_trap(wave, 10.0, 0.2)
    wf_trap1 = proc.sub_wave(wf_trap, 10000)


    wave = data[0]["waveform"]["values"].nda[0]
    energy = 5
    pulse, start = proc.get_pulse(energy, 60, len(wave))
    wave = wave+pulse
    wave = proc.sub_wave(wave, 10000)
    trap_energy, wf_trap2 = proc.apply_trap(wave, 7.0, 0.2)

    #energy, wf_trap = proc.apply_trap(wp, 7.0, 0.2)

    plt.plot(wf_trap1, label = "sub second")
    plt.plot(wf_trap2, label = "sub first")
    plt.title("Example Waveform Breakdown")
    plt.legend()
    plt.xlabel("us")
    plt.ylabel("adc")
    plt.show()

if __name__ == "__main__":
    main()