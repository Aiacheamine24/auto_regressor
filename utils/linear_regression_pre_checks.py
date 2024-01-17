# Externals imports
import numpy as np

def linear_regression_pre_checks (X_train, X_test, y_train, y_test):
    """
    This function checks the data for linear regression
    """
    if X_train is None or X_test is None or y_train is None or y_test is None:
        return {
            "success": False,
            "message": "Data not found"
        }
    
    if len(X_train) != len(y_train):
        return {
            "success": False,
            "message": "X_train and y_train must have the same length"
        }

    if len(X_test) != len(y_test):
        return {
            "success": False,
            "message": "X_test and y_test must have the same length"
        }
    
    if len(X_train[0]) != len(X_test[0]):
        return {
            "success": False,
            "message": "X_train and X_test must have the same number of columns"
        }
    
    if len(y_train[0]) != len(y_test[0]):
        return {
            "success": False,
            "message": "y_train and y_test must have the same number of columns"
        }
    
    if not isinstance(X_train, np.ndarray):
        x_train = np.array(X_train)

    if not isinstance(X_test, np.ndarray):
        x_test = np.array(X_test)
    
    if not isinstance(y_train, np.ndarray):
        y_train = np.array(y_train)
    
    if not isinstance(y_test, np.ndarray):
        y_test = np.array(y_test)
    
    return {
        "success": True,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test
    }