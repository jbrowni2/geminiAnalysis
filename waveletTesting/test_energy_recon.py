import processes.foundation as fd
import processes.fitModel as fM
import processes.histogramAction as hA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lib.processors as proc
import pywt

def main():
    data = fd.get_t1_data(9569, "Det1724")
    wave = data[0]["waveform"]["values"].nda[0]
    energy = 10
    pulse, start = proc.get_pulse(energy, 60, len(wave))
    wave = proc.sub_wave(wave, 6500)
    wp = wave+pulse
    energy, w_trap = proc.apply_trap(wp, 10.0, 0.2)

    #print(pywt.wavelist('rbio'))
    cDs = pywt.swt(w_trap, "rbio2.2")

    #cDs = proc.denoiseWave(cDs, len(wave))

    #cDs2 = proc.haar_wave(wave, level=2)

    #wave = pywt.iswt(cDs, "Haar")

    #energy, wf_trap = proc.apply_trap(wp, 7.0, 0.2)

    plt.plot(cDs[0][1][0:15000]**2)
    #plt.plot(w_trap[0:15000])
    #plt.plot(wp)
    #plt.plot(wf_trap)
    #plt.title("Example Waveform Breakdown")
    #plt.xlabel("us")
    #plt.ylabel("adc")
    plt.show()

if __name__ == "__main__":
    main()