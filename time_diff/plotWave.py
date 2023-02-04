import processes.foundation as fd
import processes.fitModel as fM
import processes.histogramAction as hA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lib.processors as proc

def main():
    data = fd.get_t1_data(9566, "Det61")
    veto = fd.get_t1_data(9566, "veto61")
    wave = data[0]["waveform"]["values"].nda[0]
    vetoWave = veto[0]["waveform"]["values"].nda[2]
    energy = 500
    pulse, start = proc.get_pulse(energy, 60, len(wave))
    wp = wave+pulse
    wp = proc.sub_wave(wp, 6500)
    x = np.asarray([x*.008 for x in range(0,len(wave))])

    #energy, wf_trap = proc.apply_trap(wp, 7.0, 0.2)

    plt.plot(wave)
    #plt.plot(wf_trap)
    plt.title("Example Waveform Breakdown")
    plt.xlabel("us")
    plt.ylabel("adc")
    plt.show()

if __name__ == "__main__":
    main()