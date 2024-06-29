from sklearnex import patch_sklearn
patch_sklearn()
from sklearn.svm import SVR

from utils import print_RMSE_MAE, save_model, import_dataset_medellin

X_train_all_data, X_test_all_data, X_val_all_data, y_train_all_data, y_test_all_data, y_val_all_data,\
X_train_relevant, X_test_relevant, X_val_relevant, y_train_relevant, y_test_relevant, y_val_relevant = \
    import_dataset_medellin()[:12]

############  SIGMOID   ############

# ALL DATA
regressor_all = SVR(kernel='sigmoid')
regressor_all.fit(X_train_all_data, y_train_all_data)

result = regressor_all.predict(X_val_all_data)
print("Sigmoid_all:")
print_RMSE_MAE(y_val_all_data, result)
save_model(regressor_all, 'medellin_svr_sigmoid_all.sav')

# RELEVANT
regressor_relevant = SVR(kernel='sigmoid')
regressor_relevant.fit(X_train_relevant, y_train_relevant)

result = regressor_relevant.predict(X_val_relevant)
print("Sigmoid_relevant:")
print_RMSE_MAE(y_val_relevant, result)
save_model(regressor_relevant, 'medellin_svr_sigmoid_relevant.sav')

############  RBF   ############

# ALL DATA
regressor_all = SVR(kernel='rbf')
regressor_all.fit(X_train_all_data, y_train_all_data)

result = regressor_all.predict(X_val_all_data)
print("RBF_all:")
print_RMSE_MAE(y_val_all_data, result)
save_model(regressor_all, 'medellin_svr_rbf_all.sav')

# RELEVANT
regressor_relevant = SVR(kernel='rbf')
regressor_relevant.fit(X_train_relevant, y_train_relevant)

result = regressor_relevant.predict(X_val_relevant)
print("RBF_relevant:")
print_RMSE_MAE(y_val_relevant, result)
save_model(regressor_relevant, 'medellin_svr_rbf_relevant.sav')

############  POLY   ############

# ALL DATA
# regressor_all = SVR(kernel='poly')
regressor_all.fit(X_train_all_data, y_train_all_data)

result = regressor_all.predict(X_val_all_data)
print("Poly_all:")
print_RMSE_MAE(y_val_all_data, result)
save_model(regressor_all, 'medellin_svr_poly_all.sav')

# RELEVANT
regressor_relevant = SVR(kernel='poly')
regressor_relevant.fit(X_train_relevant, y_train_relevant)

result = regressor_relevant.predict(X_val_relevant)
print("Poly_relevant:")
print_RMSE_MAE(y_val_relevant, result)
save_model(regressor_relevant, 'medellin_svr_poly_relevant.sav')

############  LINEAR   ############

# ALL DATA
regressor_all = SVR(kernel='linear')
regressor_all.fit(X_train_all_data, y_train_all_data)

result = regressor_all.predict(X_val_all_data)
print("Linear_all:")
print_RMSE_MAE(y_val_all_data, result)
save_model(regressor_all, 'medellin_svr_linear_all.sav')

# RELEVANT
regressor_relevant = SVR(kernel='linear')
regressor_relevant.fit(X_train_relevant, y_train_relevant)

result = regressor_relevant.predict(X_val_relevant)
print("Linear_relevant:")
print_RMSE_MAE(y_val_relevant, result)
save_model(regressor_relevant, 'medellin_svr_linear_relevant.sav')
