import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from scipy import stats
from scipy import signal
from math import exp
import processes.foundation as fd
from math import exp, sqrt, pi, erfc
from lmfit import Model
import csv

def main():
    
    waves = fd.get_t1_data(8326, "Det1726")
    #HV = fd.get_t1_data(8231, "HV1")
    #print(HV[0]["voltage"])
    wave = waves[0]["waveform"]["values"].nda[6]
    print(len(wave))
    x = np.asarray([x for x in range(0,len(wave))])
    x = x*.008
    
    plt.plot(x, wave)
    #plt.axvline(0, color = 'r')
    #plt.axvline(147, color = 'r')
    #plt.axvline(148, color = 'g')
    #plt.axvline(295, color = 'g')
    plt.xlabel("Time [us]")
    plt.ylabel("ADC")
    plt.title('Waveform while voltage was "0"')
    #plt.text(40, 38690, "Coincidence", fontsize=12)
    #plt.text(180, 38690, "AntiCoincidence", fontsize=12)
    #plt.ylim(38580, 38700)
    plt.show()



if __name__ == "__main__":
    main()