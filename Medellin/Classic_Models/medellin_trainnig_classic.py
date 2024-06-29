from utils import print_RMSE_MAE, save_model, import_dataset_medellin
import numpy as np
from math import pi
from scipy.optimize import curve_fit


def log_distance_path_loss(distance, pl0, n):
    return pl0 + 10 * n * np.log10(distance)


X_train_all_data, X_test_all_data, X_val_all_data, y_train_all_data, y_test_all_data, y_val_all_data,\
    X_train_relevant, X_test_relevant, X_val_relevant, y_train_relevant, y_test_relevant, y_val_relevant,\
    scaler_all, scaler_sig = import_dataset_medellin()[:14]

X_train_nao_norm = scaler_all.inverse_transform(X_train_all_data)
X_val_nao_norm = scaler_all.inverse_transform(X_val_all_data)

# Free Space
print("Free Space:")
result = 20* np.log10((4*pi*X_val_nao_norm[:,0])/(299.792458/905))
print_RMSE_MAE(y_val_all_data, result)

# Log Distance
print("\nLog Distance:")
popt, pcov = curve_fit(log_distance_path_loss, X_train_nao_norm[:,0]/1000, y_train_all_data)
result = log_distance_path_loss(X_val_nao_norm[:,0]/1000, popt[0], popt[1])
result_oulu = log_distance_path_loss(X_val_nao_norm[:,0]/1000, 128.95, 2.32)
print("Fit:")
print_RMSE_MAE(y_val_all_data, result)

print("Oulu:")
print_RMSE_MAE(y_val_all_data, result_oulu)

# Okumura-Hata
print("\nOkumura-Hata:")
ahr = (1.1*np.log10(905) - 0.7)*(2) - (1.56*np.log10(905) - 0.8)
result = 69.55 + 26.16 * np.log10(905) - 13.82*np.log10(X_val_nao_norm[:,2]) - ahr + \
         (44.9 - 6.55*np.log10(X_val_nao_norm[:,2]))*np.log10(X_val_nao_norm[:,0]/1000)

print_RMSE_MAE(y_val_all_data, result)

# COST231-Hata
print("\nCOST231-Hata:")
A = 46.3 + 33.9*np.log10(905) - 13.28* np.log10(X_val_nao_norm[:,2]) - ahr
B = 44.9 - 6.55*np.log10(2)
C=0

result = A + B*np.log10(X_val_nao_norm[:,0]/1000)+C
print_RMSE_MAE(y_val_all_data, result)

# Egli
print("\nEgli:")
beta = (40/905)**2
result = -10*np.log10(beta * (X_val_nao_norm[:,2]*2/(X_val_nao_norm[:,0])**2)**2)
print_RMSE_MAE(y_val_all_data, result)