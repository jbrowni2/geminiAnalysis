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

energy_df = pd.read_csv(r'baseVsNoiseCSV1723SubWave/energyArr' + str(4000) + '.csv')
energy_df = energy_df[energy_df["trapEmax"]<200]

plt.hist(energy_df["trapEmax"], histtype="step", bins=1000)
plt.show()