import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


class Circuit:
    def __init__(self, circuit):
        # circuit = 'r-[r,cpe]-[r-cpe,cpe]'
        circuit = circuit.replace(' ', '')
        self.circuit = circuit

        circuit_func = build_circuit(self.circuit)
        self.circuit_func = circuit_func

    def fit(self, frequency, impedance):
        parameters = fit_circuit(frequency, impedance)
        self.parameters = parameters
        return self


def build_circuit(circuit):
    # circuit = 'l-r-(r,cpe)-(r-cpe,cpe)'
    # circuit = 'l-(cpe,(cpe,r)-r)-r-(cpe,r)'
    
    def circuit_func(frequency, *parameters):
        omega = 2*np.pi*frequency
        z = omega*parameters[0]
        z_real = np.real(z)
        z_imag = np.imag(z)
        return np.stack([z_real, z_imag], 1)
    return circuit_func


def fit_circuit(frequency, impedance):
    parameters, _ = curve_fit(frequency, impedance)
    return parameters
