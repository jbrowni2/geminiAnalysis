import sys
import lib.definition as proc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px

import processes.foundation as fd
import processes.fitModel as fM
import processes.histogramAction as hA

from lmfit import Model


def main():
    energy_df = pd.read_csv(r'baseVsEnergyCSV1723/energyArr8000.csv')
    energy_df = energy_df[energy_df["trapEmax"]<20]
    energy_df["cal_energy"] = energy_df["trapEmax"]*0.09976049794-0.07980839835
    energy_df["true_energy"] = 5.812*0.09976049794-0.07980839835

    plt.hist(energy_df['cal_energy'], histtype='step', bins=100)
    plt.show()

if __name__ == "__main__":
    main()