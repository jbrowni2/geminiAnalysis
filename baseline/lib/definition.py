import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy import stats
from scipy import signal
from math import exp
import processes.foundation as fd
from math import exp, sqrt, pi, erfc
from lmfit import Model
import csv
from scipy.optimize import curve_fit
from pywt import swt, iswt
from statistics import median
import copy
from matplotlib import colors
import h5py
from numba import jit



def getWaves(path):
    f = h5py.File(path, 'r')
    waves = np.asarray(f["Card1/waveform/values"])
    return waves

@jit(nopython=True)
def get_pulse(energy, time, length):
    rise = time
    energy = energy



    pulse = np.zeros(length)
    x = np.linspace(-rise-40,rise+40,2*rise+81)
    y = energy/(1 + np.exp(-x/(0.3*rise)))
    start_time = 12000

    pulse[12000: 12000+len(x)] = y
    pulse[12000+len(x)::] = y[-1]


    return pulse, start_time


@jit(nopython=True)
def find_idx(arr, val, idxBegin):
        for i in range(idxBegin-1, 0, -1):
            count = arr[i]
            if count <= val:
                break

        idx = i
            
        return idx

@jit(nopython=True)
def find_idxr(arr, val, idxBegin):
        for i in range(idxBegin+1, len(arr)-1, 1):
            count = arr[i]
            if count >= val:
                break
        
        idx = i

        return idx

@jit(nopython=True)
def sub_wave(wave, length):
    mean = np.nan
    stdev = np.nan
    slope = np.nan
    intercept = np.nan


    sum_x = sum_x2 = sum_xy = sum_y = mean = stdev = 0
    isum = length

    for i in range(0, length, 1):
    # the mean and standard deviation
        temp = wave[i] - mean
        mean += temp / (i + 1)
        stdev += temp * (wave[i] - mean)

        # linear regression
        sum_x += i
        sum_x2 += i * i
        sum_xy += wave[i] * i
        sum_y += wave[i]

    slope = (isum * sum_xy - sum_x * sum_y) / (isum * sum_x2 - sum_x * sum_x)
    intercept = (sum_y - sum_x * slope) / isum

    line = np.array([x * slope + intercept for x in range(0, len(wave))])
    wave_sub = wave - line

    return wave_sub


#rise and flat are in micro seconds
@jit(nopython=True)
def apply_trap(wp, rise, flat):
    w_trap = np.zeros(len(wp))

    rise = int(rise/.008)
    flat = int(flat/.008)

    w_trap[0] = wp[0]/rise
    for i in range(1, rise, 1):
        w_trap[i] = w_trap[i - 1] + wp[i] / rise
    for i in range(rise, rise + flat, 1):
        w_trap[i] = w_trap[i - 1] + (wp[i] - wp[i - rise])/rise
    for i in range(rise + flat, 2 * rise + flat, 1):
        w_trap[i] = w_trap[i - 1] + (wp[i] - wp[i - rise] - wp[i - rise - flat])/rise
    for i in range(2 * rise + flat, len(wp), 1):
        w_trap[i] = (
        w_trap[i - 1]
        + (wp[i]
        - wp[i - rise]
        - wp[i - rise - flat]
        + wp[i - 2 * rise - flat])/rise
    )

    return np.max(w_trap), w_trap


@jit(nopython=True)
def asym_trap(w_in, rise, flat, fall):
    w_out = np.zeros(len(w_in))

    rise = int(rise/.008)
    flat = int(flat/.008)
    fall = int(fall/.008)


    w_out[0] = w_in[0] / rise
    for i in range(1, rise, 1):
        w_out[i] = w_out[i - 1] + w_in[i] / rise
    for i in range(rise, rise + flat, 1):
        w_out[i] = w_out[i - 1] + (w_in[i] - w_in[i - rise]) / rise
    for i in range(rise + flat, rise + flat + fall, 1):
        w_out[i] = (
            w_out[i - 1]
            + (w_in[i] - w_in[i - rise]) / rise
            - w_in[i - rise - flat] / fall
        )
    for i in range(rise + flat + fall, len(w_in), 1):
        w_out[i] = (
            w_out[i - 1]
            + (w_in[i] - w_in[i - rise]) / rise
            - (w_in[i - rise - flat] - w_in[i - rise - flat - fall]) / fall
        )


    trap_time = find_idx(w_out, 0.1*np.max(w_out), np.argmax(w_out))

    return w_out, np.max(w_out), trap_time

@jit(nopython=True)
def cal_sig(arr):
    #Here is where I calculate the variance
    sig = np.std(arr)
    err = sig/(np.sqrt(2*len(arr)-2))

    return sig, err

@jit(nopython=True)
def denoiseWave(cDs, Length):

    j = 0
    threshold = []
    for cD in cDs:
        median_value = np.median(cD[1])
        median_average_deviation = np.median(np.absolute(cD[1] - median_value))
        #median_average_deviation = np.median([np.absolute(number-median_value) for number in cD[1]])
        sig1 = median_average_deviation/0.6745
        threshold.append(np.float64(sig1*np.sqrt(2*np.log(Length))))

    j = 0
    for cD in cDs:
        cD[1][np.absolute(cD[1]) < threshold[j]] = np.float64(0.0)
        j += 1

    return cDs