import processes.foundation as fd
import processes.fitModel as fM
import processes.histogramAction as hA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lib.processors as proc
from lmfit import Model

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

def main():
    df= fd.get_df(9569, "Det1723")
    sig, sig_err, fw, fw_err, result, xdata = noiseFit(df["trapEmax"])
    print(sig)
    plt.close()
    plt.hist(df["trapEmax"], histtype="step", bins=1000000)
    plt.plot(xdata, result.best_fit)
    plt.xlim(0,20)
    plt.show()

if __name__=="__main__":
    main()