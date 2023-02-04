import processes.foundation as fd
import processes.fitModel as fM
import processes.histogramAction as hA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lib.processors as proc
from lmfit import Model
from numba import jit
import plotly.graph_objects as go

def noiseFit(energy_arr):

    counts, bins, bars = plt.hist(energy_arr, histtype='step', bins=1000000)

    lower = hA.find_nearest_bin(bins, -2)
    upper = hA.find_nearest_bin(bins, 4)
    ydata = counts[lower:upper]
    xdata = bins[lower:upper]


    
    gmodel = Model(fM.lingaus)
    #gmodel = Model(gaus)
    i = np.argmax(ydata)
    #params = gmodel.make_params(A=700, m1=315.5, s1=0.5, H_tail=-0.000001, H_step=1, tau=-0.5, slope=-6, intrcpt=180)
    params = gmodel.make_params(a1=550, m1=5, s1=2, slope=0.0, intrcpt=0.0)
    #params['s1'].vary = False
    result = gmodel.fit(ydata,params, x=xdata)

    sigma1 = result.params['s1'].value
    fw1 = 2.355*sigma1
    err = result.params['s1'].stderr
    err1 = err*2.355
    energy = result.params['m1'].value

    return sigma1, err, fw1, err1, result, xdata

@jit(nopython=True)
def find_idx(time, tArr2, const):
    i=0
    for t2 in tArr2:
        if np.absolute(time + 5857 - t2 + 500 - const) > 100000:
            i+=1
        else:
            return i
    
    return -1

def main():
    offset = 13389160268

    df1= fd.get_t1_data(9566, "Det61")
    df2 = fd.get_t1_data(9566, "veto61")
    times = df1[0]["timestamp"].nda
    times2 = df2[0]["timestamp"].nda
    waves = df1[0]["waveform"]["values"].nda
    wf_min = np.min(waves, axis=1)
    times = times[wf_min < 5000]

    time_diff_idx = np.zeros(len(times))
    time_diff = np.zeros(len(times))
    for i,time in enumerate(times2):
        time_diff_idx[i] = find_idx(time, times, offset)

    j=0
    for idx in time_diff_idx:
        if idx == -1:
            print(j)
            j+=1
            continue
        if j != 0 and idx == 0:
            break
        time_diff[j] = times2[j] + 500 - times[int(idx)] + 5857 - offset
        j += 1


    print("time idx", time_diff_idx[3000:3020])
    print("time diffs",time_diff[3000:3020])

    x = np.asarray([x for x in range(0,len(time_diff))])


    fig = go.Figure()

    fig.add_trace(go.Scatter(x=x,y=time_diff, name = "DeltaT"))
    fig.update_xaxes(title="Index of Veto Panel Event 61 [#]", title_font_size=40)
    fig.update_yaxes(title="(T_Veto - T_det - offset) [ns]", title_font_size=40)

    fig.update_layout(
        title="Timestamp Difference Over 10 minutes",
        title_font_size=40,
        font=dict(
            family="Courier New, monospace",
            size=20
        )
    )
    fig.show()
    

if __name__=="__main__":
    main()