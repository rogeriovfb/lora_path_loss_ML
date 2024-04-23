from math import pi
import numpy as np
from scipy.optimize import curve_fit


def path_loss_free_space(distance):
    # Friis equation
    result = 20 * np.log10((4 * pi * distance[:, 0]) / (299.792458 / 905))
    return result


def path_loss_egli(distance):
    beta = (40 / 905) ** 2
    result = -10 * np.log10(beta * (distance[:, 2] * 2 / (distance[:, 0]) ** 2) ** 2)
    return result


def path_loss_okumura_hata(distance):
    ahr = (1.1 * np.log10(905) - 0.7) * (2) - (1.56 * np.log10(905) - 0.8)
    result = 69.55 + 26.16 * np.log10(905) - 13.82 * np.log10(distance[:, 2]) - ahr + (
                44.9 - 6.55 * np.log10(distance[:, 2])) * np.log10(distance[:, 0]/1000)
    return result


def path_loss_cost231_hata(distance):
    ahr = (1.1 * np.log10(905) - 0.7) * (2) - (1.56 * np.log10(905) - 0.8)
    A = 46.3 + 33.9 * np.log10(905) - 13.28 * np.log10(distance[:, 2]) - ahr
    B = 44.9 - 6.55 * np.log10(2)
    C = 0
    result = A + B * np.log10(distance[:, 0]/1000) + C
    return result


def log_distance_path_loss(distance, pl0, n):
    return pl0 + 10 * n * np.log10(distance)


def path_loss_log_distance(X_train, y_train, distance):
    popt, pcov =  curve_fit(log_distance_path_loss, X_train[:, 0]/1000, y_train)
    result = log_distance_path_loss(distance[:, 0]/1000, popt[0], popt[1])
    result_oulu = log_distance_path_loss(distance[:, 0]/1000, 128.95, 2.32)
    return result, result_oulu
