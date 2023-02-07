import sys
import lib.definition as proc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import processes.foundation as fd
import processes.fitModel as fM
import processes.histogramAction as hA

from lmfit import Model


def fit_energy(energy_df):
    counts, bins, bars = plt.hist(energy_df['cal_energy'], histtype='step', bins=1000)

    i = np.argmax(counts)

    lower = hA.find_nearest_bin(bins, bins[i]-5)
    upper = hA.find_nearest_bin(bins, bins[i]+5)
    ydata = counts[lower:upper]
    xdata = bins[lower:upper]


    
    gmodel = Model(fM.lingaus)
    #gmodel = Model(gaus)
    i = np.argmax(ydata)
    #params = gmodel.make_params(A=700, m1=315.5, s1=0.5, H_tail=-0.000001, H_step=1, tau=-0.5, slope=-6, intrcpt=180)
    params = gmodel.make_params(a1=1700, m1=xdata[i], s1=1.5, slope=0.0, intrcpt=0.0)
    #params['s1'].vary = False
    result = gmodel.fit(ydata,params, x=xdata)

    sigma1 = result.params['s1'].value
    fw1 = 2.355*sigma1
    err = result.params['s1'].stderr
    err1 = err*2.355
    energy = result.params['m1'].value
    print(fw1)

    return sigma1, err, np.abs(fw1), err1, result.params['m1'].value

def main():
    base_min = 1800
    base_max = 10000
    base_lis = np.asarray([x for x in np.arange(base_min, base_max, 200)])
    l=0
    sig_lis1 = []
    fw_lis1 = []
    sig_error_lis1 = []
    fw_error_lis1 = []
    energy_lis1 = []
    sig_lis2 = []
    fw_lis2 = []
    sig_error_lis2 = []
    fw_error_lis2 = []
    energy_lis2 = []
    sig_lis3 = []
    fw_lis3 = []
    sig_error_lis3 = []
    fw_error_lis3 = []
    energy_lis3 = []
    sig_lis4 = []
    fw_lis4 = []
    sig_error_lis4 = []
    fw_error_lis4 = []
    energy_lis4 = []
    for base_line in base_lis:
        energy_df = pd.read_csv(r'baseVsNoiseCSV1723SubWave/energyArr' + str(base_line) + '.csv')
        energy_df = energy_df[energy_df["trapEmax"]<200]
        energy_df["cal_energy"] = energy_df["trapEmax"]
        energy_df["true_energy"] = 5.812
        energy_df["base_line"] = base_line

        sig, sig_error, fw, fw_error, energy = fit_energy(energy_df)
        sig_lis1.append(sig)
        sig_error_lis1.append(sig_error)
        fw_lis1.append(fw)
        fw_error_lis1.append(fw_error)
        energy_lis1.append(energy-(5.812))


    for base_line in base_lis:
        energy_df = pd.read_csv(r'baseVsNoiseCSV1723SubTrap/energyArr' + str(base_line) + '.csv')
        energy_df = energy_df[energy_df["trapEmax"]<20]
        energy_df["cal_energy"] = energy_df["trapEmax"]
        energy_df["true_energy"] = 5.812
        energy_df["base_line"] = base_line

        sig, sig_error, fw, fw_error, energy = fit_energy(energy_df)
        sig_lis2.append(sig)
        sig_error_lis2.append(sig_error)
        fw_lis2.append(fw)
        fw_error_lis2.append(fw_error)
        energy_lis2.append(energy-(5.812))
    
    for base_line in base_lis:
        energy_df = pd.read_csv(r'baseVsNoiseCSV1723WaveletSubWave/energyArr' + str(base_line) + '.csv')
        energy_df = energy_df[energy_df["trapEmax"]<20]
        energy_df["cal_energy"] = energy_df["trapEmax"]
        energy_df["true_energy"] = 5.812
        energy_df["base_line"] = base_line

        sig, sig_error, fw, fw_error, energy = fit_energy(energy_df)
        sig_lis3.append(sig)
        sig_error_lis3.append(sig_error)
        fw_lis3.append(fw)
        fw_error_lis3.append(fw_error)
        energy_lis3.append(energy-(5.812))
    
    """
    for base_line in base_lis:
        energy_df = pd.read_csv(r'baseVsNoiseCSV1726/energyArr' + str(base_line) + '.csv')
        energy_df = energy_df[energy_df["trapEmax"]<20]
        energy_df["cal_energy"] = energy_df["trapEmax"]*0.09458385296-0.09174633738
        energy_df["true_energy"] = 5.812*0.09458385296-0.09174633738
        energy_df["base_line"] = base_line

        sig, sig_error, fw, fw_error, energy = fit_energy(energy_df)
        sig_lis4.append(sig)
        sig_error_lis4.append(sig_error)
        fw_lis4.append(fw)
        fw_error_lis4.append(fw_error)
        energy_lis4.append(energy-(5.812*0.09458385296-0.09174633738))
    """

    df1 = pd.DataFrame({
        "Base_line":pd.Series(base_lis),
        "sigma":pd.Series(sig_lis1),
        "sigma_error":pd.Series(sig_error_lis1),
        "FWHM":pd.Series(fw_lis1),
        "FWHM_error":pd.Series(fw_error_lis1),
        "Energy_diff":pd.Series(energy_lis1)
    })

    df2 = pd.DataFrame({
        "Base_line":pd.Series(base_lis),
        "sigma":pd.Series(sig_lis2),
        "sigma_error":pd.Series(sig_error_lis2),
        "FWHM":pd.Series(fw_lis2),
        "FWHM_error":pd.Series(fw_error_lis2),
        "Energy_diff":pd.Series(energy_lis2)
    })

    df3 = pd.DataFrame({
        "Base_line":pd.Series(base_lis),
        "sigma":pd.Series(sig_lis3),
        "sigma_error":pd.Series(sig_error_lis3),
        "FWHM":pd.Series(fw_lis3),
        "FWHM_error":pd.Series(fw_error_lis3),
        "Energy_diff":pd.Series(energy_lis3)
    })

    """
    df4 = pd.DataFrame({
        "Base_line":pd.Series(base_lis),
        "sigma":pd.Series(sig_lis4),
        "sigma_error":pd.Series(sig_error_lis4),
        "FWHM":pd.Series(fw_lis4),
        "FWHM_error":pd.Series(fw_error_lis4),
        "Energy_diff":pd.Series(energy_lis4)
    })
    """

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=df1["Base_line"], y=df1["FWHM"], name="Subtract Before Trap", 
    error_y=dict(
        type='data',
        array=np.asarray(df1["FWHM_error"]),
        visible=True
    )))
    fig.add_trace(go.Scatter(x=df2["Base_line"], y=df2["FWHM"], name="Subtract After Trap", 
    error_y=dict(
        type='data',
        array=np.asarray(df2["FWHM_error"]),
        visible=True
    )))
    fig.add_trace(go.Scatter(x=df3["Base_line"], y=df3["FWHM"], name="Wavelet Filtered Subtract ", 
    error_y=dict(
        type='data',
        array=np.asarray(df3["FWHM_error"]),
        visible=True
    )))
    """
    fig.add_trace(go.Scatter(x=df4["Base_line"], y=df4["FWHM"], name="Det1726", 
    error_y=dict(
        type='data',
        array=np.asarray(df4["FWHM_error"]),
        visible=True
    )))
    """

    """
    fig.add_vline(x=8500, annotation_text="Max Baseline 9 Channels", annotation_font_size=20)
    fig.add_vline(x=15000, annotation_text="Max Baseline 5 Channels", annotation_font_size=20)
    fig.update_xaxes(title="Length of BaseLine [Clocks]", title_font_size=20)
    fig.update_yaxes(title="FWHM [keV]", title_font_size=20)
    """
    fig.update_xaxes(title="Length of BaseLine [Clocks]", title_font_size=20)
    fig.update_yaxes(title="FWHM [ADC]", title_font_size=20)

    fig.update_layout(
        title="FWHM Vs Length of Baseline For Different Methods",
        title_font_size=20,
        font=dict(
        family="Courier New, monospace",
        size=20
    )
    )
    fig.show()



if __name__=="__main__":
    main()