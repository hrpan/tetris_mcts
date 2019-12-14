import numpy as np
from math import gamma, lgamma, exp, sqrt, log
from numba import vectorize

sqrt_2pi = np.sqrt(2 * 3.1415926)
sqrt_pi = np.sqrt(3.1415926)


@vectorize(nopython=True)
def std_quantile(nu, t):
    """
    student t distribution quantile approximation from
    https://www.econstor.eu/bitstream/10419/29554/1/612504409.pdf
    """

    nu_half = nu / 2.0
    factor = lgamma(nu_half - 0.5) - lgamma(nu_half)
    factor = exp(factor)
    """
    nomin = pow(nu_half, nu_half)
    denom = gamma(nu_half) * sqrt_2pi * pow(2.0, 0.5 - nu_half)
    """
    nomin = pow(nu, nu_half)
    denom = 2 * sqrt_pi
    c = nomin / denom * factor

    return pow(c * t, 1 / nu)


@vectorize(nopython=True)
def std_quantile2(nu, t):
    """
    A Simple Approximation for the Percentiles of the t Distribution
    Author(s): Kenneth J. Koehler
    Source:
    Technometrics,
     Vol. 25, No. 1 (Feb., 1983), pp. 103-105
    Published by: Taylor & Francis, Ltd. on behalf of American Statistical Association and
    American Society for Quality
    Stable URL: https://www.jstor.org/stable/1267732
    """
    alpha = 2 / t

    f_nu = 1 / (nu + 1)

    g_alpha = 1 / sqrt(- log(alpha * (2 - alpha)))

    h_nu_alpha = pow(2 * alpha * sqrt(nu), 1 / nu)

    t_inverse = -0.0953 - 0.631 * f_nu + 0.81 * g_alpha + 0.076 * h_nu_alpha

    return 1 / t_inverse


@vectorize(nopython=True)
def norm_quantile(t):
    """
    Based on
    "Very Simply Explicitly Invertible Approximations ofNormal Cumulative and Normal Quantile Function"
    http://m-hikari.com/ams/ams-2014/ams-85-88-2014/epureAMS85-88-2014.pdf
    """
    alpha = 1 - 1 / t

    q = 10 * log(1 - log(-log(alpha) / log(2)) / log(22)) / log(41)

    return q
