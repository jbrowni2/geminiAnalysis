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
    energy = 4
    pulse, start = proc.get_pulse(energy, 60, len(wave))
    wave = wave+pulse
    #wave = wave-wave[0]

    """
    cDs= pywt.swt(wave[0:15488], "haar", level=6)
    cd = proc.haar_swt(wave[0:15488], 4)
    j = 0
    threshold = []
    for cD in cDs:
        median_value = np.median(cD[1])
        median_average_deviation = np.median(np.absolute(cD[1] - median_value))
        #median_average_deviation = np.median([np.absolute(number-median_value) for number in cD[1]])
        sig1 = median_average_deviation/0.6745
        threshold.append(0.8*np.float64(sig1*np.sqrt(2*np.log(len(wave)))))

    j = 0
    for cD in cDs:
        print("pywavelet coefficients",cD[1])
        print("my coefficients",cd[2])
        cD[1][np.absolute(cD[1]) < threshold[j]] = np.float64(0.0)
        j += 1

    w_denoise = pywt.iswt(cDs, "Haar")
    """
    #print(cDsPYWT[0][0][0:5])
    #print(len(wave))
    #w_denoise = proc.haar_swt(wave[0:15488], 5)
    w_denoise = proc.denoiseWave(wave[0:15488], 5)
    #cd3, ca3 = proc.haar_wave(cd)
    #cd = proc.denoiseWave(cd, len(cd))
    #wave_iswt = proc.haar_iswt(cd, ca)

    #cDs = proc.denoiseWave(cDs, len(wave))
    #print("pywavelet cds: ",cDs[1][1][0:4])
    #print(ca[0:4])

    #cDs2 = proc.haar_wave(wave, level=2)

    #wave = pywt.iswt(cDs, "Haar")

    #energy, wf_trap = proc.apply_trap(wp, 7.0, 0.2)

    plt.plot(wave, label="Original Waveform")
    plt.plot(w_denoise, label="Filtered Waveform")
    #plt.plot(ca)
    #plt.plot(wf_trap)
    plt.title("Example of Denoising a 0.4 KeV Waveform", fontsize=20)
    plt.xlabel("clocks [8ns]", fontsize=20)
    plt.ylabel("Current [adc]", fontsize=20)
    plt.legend(fontsize=12)
    plt.show()

if __name__ == "__main__":
    main()