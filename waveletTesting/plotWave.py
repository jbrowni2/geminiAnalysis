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
    energy = 5
    pulse, start = proc.get_pulse(energy, 60, len(wave))
    wave = wave+pulse

    cDs = pywt.swt(wave[0:15000], "haar", level=6)
    #print(len(wave))
    #cd, ca = proc.haar_wave(wave[0:15000])
    #cd = proc.denoiseWave(cd, len(cd))
    #wave_iswt = proc.haar_iswt(cd, ca)

    #cDs = proc.denoiseWave(cDs, len(wave))
    #print("pywavelet cds: ",cDs[1][1][0:4])
    #print(ca[0:4])

    #cDs2 = proc.haar_wave(wave, level=2)

    #wave = pywt.iswt(cDs, "Haar")

    #energy, wf_trap = proc.apply_trap(wp, 7.0, 0.2)

    plt.plot(wave)
    #plt.plot(wave_iswt)
    #plt.plot(ca)
    #plt.plot(wf_trap)
    #plt.title("Example Waveform Breakdown")
    #plt.xlabel("us")
    #plt.ylabel("adc")
    plt.show()

if __name__ == "__main__":
    main()