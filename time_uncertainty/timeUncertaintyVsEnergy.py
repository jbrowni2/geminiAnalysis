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


def get_energies(waves, energy):
    
    #pulse, start = proc.get_pulse(energy, 60, len(waves["waveform"]["values"].nda[0]))
    #waves["waveform"]["values"].nda = np.asarray([wave + pulse for wave in waves["waveform"]["values"].nda])
    #sub = np.vectorize(proc.sub_wave, signature="(n),()->(n)")
    #trap = np.vectorize(proc.apply_trap, signature="(n),(),()->()")
    energy_arr = np.zeros(len(waves))
    cal_t50 = np.zeros(len(waves))
    cal_t90 = np.zeros(len(waves))
    cal_t10 = np.zeros(len(waves))


    pulse, start = proc.get_pulse(energy, 60, 15488)
    for j,wave in enumerate(waves):
        wave = proc.denoiseWave(wave, 5)

        wave = proc.sub_wave(wave[0:15488], 5)
        #wave = np.int64(wave) - np.int64(wave[1])
        wp = wave+pulse
        trap_energy, onset = proc.apply_trap(wp[0:15000], 7.0, 0.2)


        m90 = trap_energy*0.9
        m10 = trap_energy*0.1
        m50 = trap_energy*0.5
        try:
            imax51 = proc.find_idx(wp, m50, onset)
        except:
            imax51 = proc.find_idxr(wp, m50, onset)
        
        try:
            imax9 = proc.find_idxr(wp, m90, imax51)
        except:
            imax9 = proc.find_idx(wp, m90, imax51)
        try:
            imax1 = proc.find_idx(wp, m10, imax51)
        except:
            imax1 = proc.find_idxr(wp, m10, imax51)


        cal_t50[j] = imax51
        cal_t90[j] = imax9
        cal_t10[j] = imax1
        #wf_trap = proc.sub_wave(wf_trap, base_line)
        #trap_energy = np.max(wf_trap)

        energy_arr[j] = trap_energy
    

    return energy_arr, cal_t90, cal_t50, cal_t10

    


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
    energy_min = 1
    energy_max = 20
    energy_lis = np.asarray([x for x in np.arange(energy_min, energy_max, 0.5)])
    waves = data[0]["waveform"]["values"].nda
    l = 0
    for energy in energy_lis:
        print(l/len(energy_lis))
        l+=1
        energy_Arr, t90_array, t10_array, t50_array, = get_energies(waves, energy)
        name = "timeUncVsEnergyCSV1723/energyArr" + str(energy) + ".csv"
        df = pd.DataFrame({"trapEmax": pd.Series(energy_Arr), "t90": pd.Series(t90_array), "t10": pd.Series(t10_array), "t50": pd.Series(t50_array)})
        df.to_csv(name)




if __name__ == "__main__":
    main()