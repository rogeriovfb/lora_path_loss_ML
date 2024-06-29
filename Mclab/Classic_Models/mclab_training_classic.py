from utils import print_RMSE_MAE, save_model, import_dataset_mclab
import numpy as np
from math import pi
from scipy.optimize import curve_fit

def log_distance_path_loss(distance, pl0, n):
    return pl0 + 10 * n * np.log10(distance)


X_train, X_test, X_val, y_train, y_test, y_val, scaler = import_dataset_mclab()[:7]
X_train_nao_norm = scaler.inverse_transform(X_train)
X_val_nao_norm = scaler.inverse_transform(X_val)

# Free Space
print("Free Space:")
result = 20* np.log10((4*pi*X_val_nao_norm[:,0]*1000)/(299.792458/868))
print_RMSE_MAE(y_val, result)

# Log Distance
print("\nLog Distance:")
popt, pcov = curve_fit(log_distance_path_loss, X_train_nao_norm[:,0], y_train)
result = log_distance_path_loss(X_val_nao_norm[:,0], popt[0], popt[1])
result_oulu = log_distance_path_loss(X_val_nao_norm[:,0], 128.95, 2.32)

print("Fit:")
print_RMSE_MAE(y_val, result)

print("Oulu:")
print_RMSE_MAE(y_val, result_oulu)

# Okumura-Hata
print("\nOkumura-Hata:")
ahr = (1.1*np.log10(868) - 0.7)*(2) - (1.56*np.log10(868) - 0.8)
result = 69.55 + 26.16 * np.log10(868) - 13.82*np.log10(X_val_nao_norm[:,2]) - \
         ahr + (44.9 - 6.55*np.log10(X_val_nao_norm[:,2]))*np.log10(X_val_nao_norm[:,0])

print_RMSE_MAE(y_val, result)

# COST231-Hata
print("\nCOST231-Hata:")
A = 46.3 + 33.9*np.log10(868) - 13.28* np.log10(X_val_nao_norm[:,2]) - ahr
B = 44.9 - 6.55*np.log10(2)
C = 0

result = A + B*np.log10(X_val_nao_norm[:,0])+C
print_RMSE_MAE(y_val, result)

# Egli
print("\nEgli:")
beta = (40/868)**2
result = -10*np.log10(beta * (X_val_nao_norm[:,2]*2/(X_val_nao_norm[:,0]*1000)**2)**2)
print_RMSE_MAE(y_val, result)