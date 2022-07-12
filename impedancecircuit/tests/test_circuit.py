import pandas as pd

from impedancecircuit.models.circuit import Circuit


cellname = '25C01'
datapath = './data/Cavendish/'
filename = f'{datapath}EIS_state_V_{cellname}.txt'

data = pd.read_csv(filename, delimiter='\t')
data.columns = data.columns.str.strip()

cycle = 1
data_real = data[data['cycle number'] == cycle]['Re(Z)/Ohm']
data_imag = data[data['cycle number'] == cycle]['-Im(Z)/Ohm']
data_freq = data[data['cycle number'] == cycle]['freq/Hz']


def test_circuit():
    circuit_str = 'l-r-(r,cpe)-(r-cpe,cpe)'
    circuit = Circuit(circuit_str)
