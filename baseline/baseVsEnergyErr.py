import sys
import lib.definition as proc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import h5py 
import json
import copy
from collections import OrderedDict
from lmfit import Model
import os
from pygama import flow
from pygama.raw.build_raw import build_raw
from pygama.dsp.build_dsp import build_dsp

from pygama.pargen.dsp_optimize import run_one_dsp
from pygama.pargen.dsp_optimize import run_grid
from pygama.pargen.dsp_optimize import ParGrid
from pygama.lgdo.lh5_store import LH5Store
import pygama.math.histogram as pgh
import pygama.math.peak_fitting as pgf
import multiprocessing as mp
import pywt



import processes.foundation as fd
import processes.fitModel as fM
import processes.histogramAction as hA


def get_energies(energy, waves, base_line):
    
    #pulse, start = proc.get_pulse(energy, 60, len(waves["waveform"]["values"].nda[0]))
    #waves["waveform"]["values"].nda = np.asarray([wave + pulse for wave in waves["waveform"]["values"].nda])
    #sub = np.vectorize(proc.sub_wave, signature="(n),()->(n)")
    #trap = np.vectorize(proc.apply_trap, signature="(n),(),()->()")
    energy_arr = np.zeros(len(waves))


    pulse, start = proc.get_pulse(energy, 60, len(waves[0]))
    for j,wave in enumerate(waves):
        cDs = pywt.swt(wave, "haar", level=2)

        cDs = proc.denoiseWave(cDs, len(wave))

        wave = pywt.iswt(cDs, "Haar")

        wave = proc.sub_wave(wave, base_line)
        #wave = np.int64(wave) - np.int64(wave[1])
        wp = wave+pulse
        trap_energy, wf_trap = proc.apply_trap(wp, 7.0, 0.2)
        #wf_trap = proc.sub_wave(wf_trap, base_line)
        #trap_energy = np.max(wf_trap)

        energy_arr[j] = trap_energy
    

    return energy_arr

    


def noiseFit(n, det):
    df = fd.get_df(n, det)

    df["Energy"] = df["trapEmax"]*0.09976049794 -0.07980839835
    counts, bins, bars = plt.hist(df['Energy'], histtype='step', bins=500000)

    i = np.argmax(counts)

    lower = hA.find_nearest_bin(bins, bins[i]-0.5)
    upper = hA.find_nearest_bin(bins, bins[i]+0.5)
    ydata = counts[lower:upper]
    xdata = bins[lower:upper]


    
    gmodel = Model(fM.lingaus)
    #gmodel = Model(gaus)
    i = np.argmax(ydata)
    print(xdata[i])
    #params = gmodel.make_params(A=700, m1=315.5, s1=0.5, H_tail=-0.000001, H_step=1, tau=-0.5, slope=-6, intrcpt=180)
    params = gmodel.make_params(a1=550, m1=xdata[i], s1=0.03, slope=0.0, intrcpt=0.0)
    #params['s1'].vary = False
    result = gmodel.fit(ydata,params, x=xdata)

    sigma1 = result.params['s1'].value
    fw1 = 2.355*sigma1
    err = result.params['s1'].stderr
    err1 = err*2.355
    energy = result.params['m1'].value

    return sigma1, err, fw1, err1


def main():
    data = fd.get_t1_data(9569, "Det1723")
    base_min = 1000
    base_max = 10000
    base_lis = np.asarray([x for x in np.arange(base_min, base_max, 200)])
    waves = data[0]["waveform"]["values"].nda
    l = 0
    for base_line in base_lis:
        print(l/len(base_lis))
        energy = 5
        l+=1
        energy_Arr = get_energies(energy, waves, base_line)
        name = "baseVsNoiseCSV1723WaveletSubWave/energyArr" + str(base_line) + ".csv"
        df = pd.DataFrame({"trapEmax": pd.Series(energy_Arr)})
        df.to_csv(name)




if __name__ == "__main__":
    main()