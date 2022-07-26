import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, '/home/ubuntu/seongyoon/PI/Impedance')
from impedancecircuit.models.circuit import Circuit


cellname = '25C01'
datapath = '../examples/data/Cavendish/'
filename = f'{datapath}EIS_state_V_{cellname}.txt'

data = pd.read_csv(filename, delimiter='\t')
data.columns = data.columns.str.strip()

cycle = 1
data_real = data[data['cycle number'] == cycle]['Re(Z)/Ohm'].values
data_imag = data[data['cycle number'] == cycle]['-Im(Z)/Ohm'].values
data_freq = data[data['cycle number'] == cycle]['freq/Hz'].values

impedance = np.concatenate([data_real, data_imag])


def test_circuit():
    circuit_str = 'l-r-(r,cpe)-(r-cpe)-cpe'
    # circuit = Circuit(circuit_str)
    custom_initial_guess = np.zeros(10)
    custom_initial_guess[1] = np.log(min(data_real))
    custom_initial_guess[0] = np.log(abs(data_imag[np.argmin(data_real)]) /
                                     (2*np.pi*data_freq[np.argmin(data_real)]))
    circuit = Circuit(circuit_str, custom_initial_guess)

    circuit.fit(data_freq, impedance)
    print(circuit.parameters)
    impedance_pred = circuit.predict(data_freq)
    print(impedance_pred)


test_circuit()
