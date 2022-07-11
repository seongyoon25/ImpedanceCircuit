import numpy as np


def R(par, freq):
    """resistor

    Args:
        par (_type_): _description_
        freq (_type_): _description_
    """
    Z = np.array(len(freq)*[par[0]])
    return Z


def C(par, freq):
    """capacitor

    Args:
        par (_type_): _description_
        freq (_type_): _description_
    """
    omega = 2*np.pi*np.array(freq)
    Z = 1./(par[0]*1j*omega)
    return Z


def L(par, freq):
    """inductor

    Args:
        par (_type_): _description_
        freq (_type_): _description_
    """
    omega = 2*np.pi*np.array(freq)
    Z = par[0]*1j*omega
    return Z


def CPE(par, freq):
    """CPE(constant phase element)

    Args:
        par (_type_): _description_
        freq (_type_): _description_
    """
    omega = 2*np.pi*np.array(freq)
    Z = 1./(par[0]*(1j*omega)**par[1])
    return Z


def p(elements):
    """parallel elements

    Args:
        elements (_type_): _description_
    """
    z = len(elements[0])*[0 + 0j]
    for element in elements:
        z += 1./element
    return 1./z
