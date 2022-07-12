import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from impedancecircuit.models.elements import R, C, L, CPE, p


class Circuit:
    def __init__(self, circuit):
        # circuit = 'l-r-(r,cpe)-(r-cpe,cpe)'
        circuit = circuit.replace(' ', '')
        self.circuit = circuit

        circuit_func, parameters = build_circuit(self.circuit)
        self.circuit_func = circuit_func
        self.parameters_init = np.array(parameters)

    def fit(self, frequency, impedance):
        parameters = fit_circuit(self.circuit_func, self.parameters_init,
                                 frequency, impedance)
        self.parameters = np.array(parameters)

    def predict(self, frequency):
        impedance = self.circuit_func(frequency, *self.parameters)
        return impedance


def build_circuit(circuit):
    # circuit = 'l-r-(r,cpe)-(r-cpe,cpe)'
    # circuit = 'l-(cpe,(cpe,r)-r)-r-(cpe,r)'

    # initial parameters will be determined after circuit is identified.
    parameters = np.ones(10)

    def circuit_func(frequency, *parameters):
        parameters = np.array(parameters).tolist()
        # Temporarily fixed circuit
        z = L([parameters[0]], frequency) + \
            R([parameters[1]], frequency) + \
            p([R([parameters[2]], frequency),
               CPE([parameters[3], parameters[4]], frequency)]) + \
            p([R([parameters[5]], frequency) +
               CPE([parameters[6], parameters[7]], frequency),
               CPE([parameters[8], parameters[9]], frequency)])
        z_real = np.real(z)
        z_imag = np.imag(z)
        return np.concatenate([z_real, z_imag])
    return circuit_func, parameters


def fit_circuit(circuit_func, parameters, frequency, impedance):
    parameters, _ = curve_fit(circuit_func, frequency, impedance,
                              p0=parameters)
    return parameters
