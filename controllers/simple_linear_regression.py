# # Externals imports
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

# # Internals Imports
# from utils.linear_regression_pre_checks import linear_regression_pre_checks

# def simple_linear_regression(X_train, X_test, y_train, y_test):
#     res = linear_regression_pre_checks(X_train, X_test, y_train, y_test)
#     if not res["success"]:
#         return res
    
#     X_train = res["X_train"]
#     X_test = res["X_test"]
#     y_train = res["y_train"]
#     y_test = res["y_test"]

#     # Training the Simple Linear Regression model on the Training set
#     from sklearn.linear_model import LinearRegression
#     regressor = LinearRegression()
#     regressor.fit(X_train, y_train)

#     # Predicting the Test set results
#     y_pred = regressor.predict(X_test)

#     # Regression Coefficient
#     m = regressor.coef_
#     # Regression Intercept
#     b = regressor.intercept_
#     # Regression Line
#     y = m*X_train + b

#     # Visualising the Training set results
#     plt.scatter(X_train, y_train, color = 'red')
#     plt.plot(X_train, y, color = 'blue')
#     plt.title('Salary vs Experience (Training set)')
#     plt.xlabel('Years of Experience')
#     plt.ylabel('Salary')
#     plt.show()


#     # Visualising the Test set results
#     plt.scatter(X_test, y_test, color = 'red')
#     plt.plot(X_train, y, color = 'blue')
#     plt.title('Salary vs Experience (Test set)')
#     plt.xlabel('Years of Experience')
#     plt.ylabel('Salary')
#     plt.show()
    

#     # Return the result to the user then the interface will display it
#     return {
#         "success": True,
#         "m": m,
#         "b": b,
#         "y": y,
#         "y_pred": y_pred
#     }

# Externals imports
from matplotlib import scale
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Internals Imports
from utils.linear_regression_pre_checks import linear_regression_pre_checks

def simple_linear_regression(X_train, X_test, y_train, y_test, scaler=None):
    res = linear_regression_pre_checks(X_train, X_test, y_train, y_test)
    if scaler is None:
        return {
            "success": False,
            "message": "Scaler is not provided"
        }
    if not res["success"]:
        return res
    
    X_train = res["X_train"]
    X_test = res["X_test"]
    y_train = res["y_train"]
    y_test = res["y_test"]

    # Training the Simple Linear Regression model on the Training set
    from sklearn.linear_model import LinearRegression
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = regressor.predict(X_test)

    # Regression Coefficient
    m = regressor.coef_
    # Regression Intercept
    b = regressor.intercept_
    # Regression Line
    y = m*X_train + b

    # Regression Line for the training set
    y_train_pred = m*X_train + b

    

    # Return the result to the user then the interface will display it
    return {
        "success": True,
        "m": m,
        "b": b,
        "y": y,
        "y_pred": y_pred,
        "y_train_pred": y_train_pred,
        "scaler": scaler
    }