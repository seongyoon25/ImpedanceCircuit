import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from impedance.elements import R, C, L, CPE, p


class Circuit:
    def __init__(self, circuit):
        # circuit = 'l-r-(r,cpe)-(r-cpe,cpe)'
        circuit = circuit.replace(' ', '')
        self.circuit = circuit

        circuit_func = build_circuit(self.circuit)
        self.circuit_func = circuit_func

    def fit(self, frequency, impedance):
        parameters = fit_circuit(self.circuit_func, frequency, impedance)
        self.parameters = parameters
        return self


def build_circuit(circuit):
    # circuit = 'l-r-(r,cpe)-(r-cpe,cpe)'
    # circuit = 'l-(cpe,(cpe,r)-r)-r-(cpe,r)'

    def circuit_func(frequency, *parameters):
        parameters = np.array(parameters).tolist()
        # Temporarily fixed circuit
        z = L(parameters[0], frequency) + \
            R(parameters[1], frequency) + \
            p([R(parameters[2], frequency),
               CPE(parameters[3], frequency)]) + \
            p([R(parameters[4], frequency) + CPE(parameters[5], frequency),
               CPE(parameters[6], frequency)])
        z_real = np.real(z)
        z_imag = np.imag(z)
        return np.stack([z_real, z_imag], 1)
    return circuit_func


def fit_circuit(circuit_func, frequency, impedance):
    parameters, _ = curve_fit(circuit_func, frequency, impedance)
    return parameters
