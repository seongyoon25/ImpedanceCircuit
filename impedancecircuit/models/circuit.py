import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from impedancecircuit.models.elements import R, C, L, CPE, p


class Circuit:
    def __init__(self, circuit, parameters=None):
        # circuit = 'l-r-(r,cpe)-(r-cpe)-cpe'
        circuit = circuit.replace(' ', '')
        self.circuit = circuit

        circuit_func, parameters = build_circuit(self.circuit, parameters)
        self.circuit_func = circuit_func
        self.parameters_init = np.array(parameters)

    def fit(self, frequency, impedance):
        parameters = fit_circuit(self.circuit_func, self.parameters_init,
                                 frequency, impedance)
        self.parameters = np.array(parameters)

    def predict(self, frequency):
        impedance = self.circuit_func(frequency, *self.parameters)
        return impedance


def build_circuit(circuit, parameters=None):
    # circuit = 'l-r-(r,cpe)-(r-cpe,cpe)'
    # circuit = 'l-(cpe,(cpe,r)-r)-r-(cpe,r)'
    # circuit = 'l-r-(r,cpe)-(r-cpe)-cpe'
    circuit_list = ['l', 'r', 'r', 'cpe', 'r', 'cpe', 'cpe']

    # initial parameters will be determined after circuit is identified.
    if parameters is None:
        parameters = np.ones(10)
    sigmoid_idx = np.zeros(len(parameters), dtype=int)
    sigmoid_idx[[4, 7, 9]] = 1

    def circuit_func(frequency, *parameters):
        parameters = np.array(parameters)
        parameters = np.where(sigmoid_idx,
                              sigmoid(parameters),
                              np.exp(parameters))
        # parameters[sigmoid_idx] = sigmoid(parameters[sigmoid_idx])
        # parameters[softplus_idx] = softplus(parameters[softplus_idx])
        # Temporarily fixed circuit
        z = L([parameters[0]], frequency) + \
            R([parameters[1]], frequency) + \
            p([R([parameters[2]], frequency),
               CPE([parameters[3], parameters[4]], frequency)]) + \
            p([R([parameters[5]], frequency),
               CPE([parameters[6], parameters[7]], frequency)]) + \
            CPE([parameters[8], parameters[9]], frequency)
        z_real = np.real(z)
        z_imag = -np.imag(z)
        return np.hstack([z_real, z_imag])
    return circuit_func, parameters


def fit_circuit(circuit_func, parameters, frequency, impedance):
    results = least_squares(lambda x: np.hstack([circuit_func(frequency, *x) - impedance,
                                                 np.abs(impedance).mean()*0.01*x]),
                            x0=parameters, method='lm', x_scale='jac')
    parameters = results.x
    return parameters


def sigmoid(x):
    """sigmoid function avoiding overflow

    Args:
        x (_type_): _description_

    Returns:
        _type_: _description_
    """
    return np.where(x < 0, np.exp(x) / (1.+np.exp(x)), 1./(1.+np.exp(-x)))
