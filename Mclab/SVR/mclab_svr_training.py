from sklearnex import patch_sklearn
patch_sklearn()

from utils import print_RMSE_MAE, save_model, import_dataset_mclab
from sklearn.svm import SVR

X_train, X_test, X_val, y_train, y_test, y_val = import_dataset_mclab()[:6]


############  SIGMOID   ############
regressor = SVR(kernel='sigmoid')
regressor.fit(X_train, y_train)

result = regressor.predict(X_val)
print("Sigmoid:")
print_RMSE_MAE(y_val, result)

save_model(regressor, 'mclab_svr_sigmoid.sav')


############  RBF   ############
regressor = SVR(kernel='rbf')
regressor.fit(X_train, y_train)

result = regressor.predict(X_val)
print("RBF:")
print_RMSE_MAE(y_val, result)

save_model(regressor, 'mclab_svr_rbf.sav')


############  POLY   ############
regressor = SVR(kernel='poly')
regressor.fit(X_train, y_train)

result = regressor.predict(X_val)
print("Poly:")
print_RMSE_MAE(y_val, result)

save_model(regressor, 'mclab_svr_poly.sav')


############  LINEAR   ############
regressor = SVR(kernel='linear')
regressor.fit(X_train, y_train)

result = regressor.predict(X_val)
print("Linear:")
print_RMSE_MAE(y_val, result)

save_model(regressor, 'mclab_svr_linear.sav')
