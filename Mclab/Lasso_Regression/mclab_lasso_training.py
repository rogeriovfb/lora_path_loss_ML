import numpy as np

from utils import print_RMSE_MAE, save_model, import_dataset_mclab
from sklearn.linear_model import Lasso

X_train, X_test, y_train, y_test = import_dataset_mclab()[:4]


error_list = []
max_alpha = 50

for i in np.arange(0, 5.0, 0.01):
    # Fit regression model
    regressor = Lasso(alpha=i)
    regressor.fit(X_train, y_train)

    # Predict
    result = regressor.predict(X_test)

    # RMSE
    square_error = np.square(np.subtract(y_test, result)).mean()
    rmse = np.sqrt(square_error)
    #print(rmse)
    error_list.append(rmse)


# Fit regression model
regressor = Lasso(alpha=0.01)
regressor.fit(X_train, y_train)

# Predict
result = regressor.predict(X_test)
print_RMSE_MAE(y_test,result)
save_model(regressor, 'mclab_lasso.sav')
