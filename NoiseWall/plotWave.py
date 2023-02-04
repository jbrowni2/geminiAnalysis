import processes.foundation as fd
import processes.fitModel as fM
import processes.histogramAction as hA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lib.processors as proc

def main():
    data = fd.get_t1_data(9569, "Det1724")
    wave = data[0]["waveform"]["values"].nda[0]
    energy = 500
    pulse, start = proc.get_pulse(energy, 60, len(wave))
    wp = wave+pulse
    wp = proc.sub_wave(wp, 6500)
    x = np.asarray([x*.008 for x in range(0,len(wave))])

    #energy, wf_trap = proc.apply_trap(wp, 7.0, 0.2)

    plt.plot(x, wave)
    #plt.plot(wf_trap)
    plt.title("Example Waveform Breakdown")
    plt.vlines(0, ymin = 35800, ymax =36100, colors="r")
    plt.vlines(6500*.008, ymin = 35800, ymax =36100, colors="r", label="Baseline Boundaries")
    plt.vlines(6600*.008, ymin = 35800, ymax =36100, colors="g", label="Signal Boundaries")
    plt.vlines(7750*.008, ymin = 35800, ymax =36100, colors="g")
    plt.vlines(7850*.008, ymin = 35800, ymax =36100, colors="r")
    plt.vlines(14250*.008, ymin = 35800, ymax =36100, colors="r")
    plt.vlines(14350*.008, ymin = 35800, ymax =36100, colors="b", label="Background Boundaries")
    plt.vlines(15500*.008, ymin = 35800, ymax =36100, colors="b")
    plt.legend()
    plt.xlabel("us")
    plt.ylabel("adc")
    plt.show()

if __name__ == "__main__":
    main()