import processes.foundation as fd
import processes.fitModel as fM
import processes.histogramAction as hA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lib.processors as proc
from lmfit import Model


def noiseFit(energy_arr):

    counts, bins, bars = plt.hist(energy_arr, histtype='step', bins=100000)

    lower = hA.find_nearest_bin(bins, 0)
    upper = hA.find_nearest_bin(bins, 10)
    ydata = counts[lower:upper]
    xdata = bins[lower:upper]


    
    gmodel = Model(fM.lingaus)
    #gmodel = Model(gaus)
    i = np.argmax(ydata)
    #params = gmodel.make_params(A=700, m1=315.5, s1=0.5, H_tail=-0.000001, H_step=1, tau=-0.5, slope=-6, intrcpt=180)
    params = gmodel.make_params(a1=550, m1=5, s1=1.5, slope=0.0, intrcpt=0.0)
    #params['s1'].vary = False
    result = gmodel.fit(ydata,params, x=xdata)

    sigma1 = result.params['s1'].value
    fw1 = 2.355*sigma1
    err = result.params['s1'].stderr
    err1 = err*2.355
    energy = result.params['m1'].value

    return sigma1, err, fw1, err1

def main():

    #data = fd.get_t1_data(9569, "Det1723")
    rise_min = 2.0
    rise_max = 20.0
    rise_lis = np.asarray([x for x in np.arange(rise_min, rise_max, 0.5)])
    #waves = data[0]["waveform"]["values"].nda
    fw_arr = np.zeros(len(rise_lis))
    fw_error_arr = np.zeros(len(rise_lis))
    l = 0
    i = 0
    for rise in rise_lis:
        rise = round(rise,2)
        l+=1
        name = "2123Conographs/1726Energy/energyArrRise" + str(rise) + ".csv"
        df = pd.read_csv(name)
        sig, sig_err, fw, fw_error = noiseFit(df["trapEmax"])
        print(fw)
        fw_arr[i] = fw
        fw_error_arr[i] = fw_error
        i+=1


    plt.clf()
    plt.errorbar(rise_lis, fw_arr, yerr=fw_error_arr)
    plt.show()

if __name__=="__main__":
    main()