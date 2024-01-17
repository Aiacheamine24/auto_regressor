# Externals imports
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

def data_preprocessing(path: str, target: list, test_size: float = 0.2, random_state: int = 1, 
                       strategy: str = "mean", method_for_scaling: str = "standard", 
                       categorical_features: list = None, numerical_features: list = None):
    """
    This function will get clear prepare and split the data into train and test data.
    """
    # Load the data
    data = pd.read_csv(path)
    if data.empty: 
        return {
            "success": False,
            "message": "Data not found"
        }

    # Split the data into X and y
    X = data.drop(columns=target)
    y = data[target]

    # If numerical_features is None, select all numerical columns
    if numerical_features is None:
        numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()

    # Protect the data from NaN
    imputer = SimpleImputer(missing_values=np.nan, strategy=strategy)
    X[numerical_features] = imputer.fit_transform(X[numerical_features])

    # Encode the categorical variables
    if categorical_features:
        ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), categorical_features)], remainder='passthrough')
        X = np.array(ct.fit_transform(X))

    # Encode the target variable if it's categorical
    if y.select_dtypes(include=([np.number])).columns.tolist() != []:
        y = np.array(y)
    else:
        le = LabelEncoder()
        y = le.fit_transform(y)

    # Split the data into train and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Feature Scaling
    if method_for_scaling == "standard":
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
    elif method_for_scaling == "min_max":
        sc = MinMaxScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
    else:
        return {
            "success": False,
            "message": "Please provide a valid method for scaling"
        }
    
    return {
        "success": True,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test
    }