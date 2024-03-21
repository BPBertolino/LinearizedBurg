'''
Created on 21 mar 2024
@author: Bruno Pavani Bertolino, Sofia Gemmi dos Santos
'''

import lasio # Library to read the structure ".las" typical in geosciences
import numpy as np # Mathematical library
import pandas as pd # Data analysis library

from LINBURG import LINBURG

# Examples of paths for some well data
path = '/data/'
arquivo = path + '9MA  0026D RJS.las'
nome = '9-MA-26D-RJS'
tipo_log = 'GR' # Gamma-ray well logs chosen
ordem = 64 # Selects the order for the Burg algorithm

data = lasio.read(arquivo)
well = data.df()
log = well[tipo_log]
log = log.dropna()
x = pd.Series(well.index.tolist()) # Vector of depths

# Calculates the limits of lists from the well horizons, as well as the interval between measurements.
top = min(x)
floor = max(x)
step = (floor-top)/len(x)

# Determines the limits (envelope) of the well from input data
envelope = pd.read_csv(path + 'envelope.csv',sep="\t")
envelope_floor_loc = envelope.loc[(envelope["WELL"] == nome) & (envelope["TOPO_BASE"] == "BASE")]
envelope_floor = envelope_floor_loc["DEPTH"]
envelope_top_loc = envelope.loc[(envelope["WELL"] == nome) & (envelope["TOPO_BASE"] == "TOPO")]
envelope_top = envelope_top_loc["DEPTH"]
base = envelope_floor
topo = envelope_top
index_base = int(round(((base - x.iloc[0])/(x.iloc[-1] - x.iloc[0])) * len(x)))
index_topo = int(round(((topo - x.iloc[0])/(x.iloc[-1] - x.iloc[0])) * len(x)))

# Calls LINBURG to perform L1 regularization and then use the Burg algorithm to obtain the prediction vectors
linburg_log = LINBURG(x, log, 100, arquivo, topo, base, ordem, tipo_log, top, step)
