import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


class Circuit:
    def __init__(self, circuit) -> None:
        
        self.circuit = circuit
        return self

    def fit(self, frequency, impedance):
        parameters = fit_circuit(frequency, impedance)
        self.parameters = parameters
        return self


def fit_circuit(frequency, impedance):
    parameters, _ = curve_fit(frequency, impedance)
    return parameters
