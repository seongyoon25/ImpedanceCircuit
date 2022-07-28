import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, '/home/ubuntu/seongyoon/PI/Impedance')
from impedancecircuit.models.circuit import Circuit, sigmoid


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
    custom_initial_guess = np.ones(10)
    custom_initial_guess[1] = np.log(min(data_real))
    custom_initial_guess[0] = np.log(abs(data_imag[np.argmin(data_real)]) /
                                     (2*np.pi*data_freq[np.argmin(data_real)]))
    circuit = Circuit(circuit_str, custom_initial_guess)

    circuit.fit(data_freq, impedance)

    sigmoid_idx = np.zeros(len(circuit.parameters), dtype=int)
    sigmoid_idx[[4, 7, 9]] = 1
    print(pd.Series(np.where(sigmoid_idx,
                             sigmoid(circuit.parameters),
                             np.exp(circuit.parameters)),
                    index=['L', 'R0', 'R1', 'Q1', 'a1', 'R2', 'Q2', 'a2', 'Q3', 'a3']))
    impedance_pred = circuit.predict(data_freq)
    print(impedance_pred)


test_circuit()
