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
import seaborn as sns
import plotly.express as px

from pygama.pargen.dsp_optimize import run_one_dsp
from pygama.pargen.dsp_optimize import run_grid
from pygama.pargen.dsp_optimize import ParGrid
from pygama.lgdo.lh5_store import LH5Store
import pygama.math.histogram as pgh
import pygama.math.peak_fitting as pgf
import multiprocessing as mp



import processes.foundation as fd
import processes.fitModel as fM
import processes.histogramAction as hA


def main():
    base_min = 1000
    base_max = 16000
    base_lis = np.asarray([x for x in np.arange(base_min, base_max, 200)])
    l=0
    lis=[]
    for base_line in base_lis:
        energy_df = pd.read_csv(r'baseVsEnergyCSV1725/energyArr' + str(base_line) + '.csv')
        energy_df = energy_df[energy_df["trapEmax"]<20]
        energy_df["cal_energy"] = energy_df["trapEmax"]*0.09974069813-0.07779774454
        energy_df["true_energy"] = 5.812*0.09974069813-0.07779774454
        energy_df["base_line"] = base_line
        lis.append(energy_df)

    df = pd.concat(lis)
    fig = px.density_heatmap(df, x="base_line", y="cal_energy", nbinsx=len(base_lis)+1, nbinsy=300
    , title="2d Histogram of Energy Reconstruction Vs Baseline For 1725")
    fig.update_xaxes(title="Length of BaseLine [Clocks]", title_font_size=20)
    fig.update_yaxes(title="Energy [keV]", title_font_size=20)
    """
    fig.add_annotation(x=0.9, y=0.15,
            text="Begins To see Pulsar Response",
            showarrow=True,
            arrowhead=1,
            font=dict(
                color="white",
                size=12
            ),
            arrowcolor="white")
    fig.add_annotation(x=0.9, y=0.4,
            text="ADC to Energy Conversion is Energy = ADC*0.09458385296 -0.09174633738",
            showarrow=False,
            font=dict(
                color="white",
                size=12
            )
            )
    """
    fig.show()
        
        




if __name__ == "__main__":
    main()