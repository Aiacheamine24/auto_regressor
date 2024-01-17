# Externals imports
import matplotlib.pyplot as plt

# Internals Imports
from controllers.data_preprecessing import data_preprocessing
from controllers.simple_linear_regression import simple_linear_regression

path = "public/files/imperfect_data_set copy.csv"

# Split the data into train and test data
result = data_preprocessing(path=path, target=["Salary"], test_size=0.05, random_state=1, 
                            strategy="mean", method_for_scaling="standard", 
                            categorical_features=None, numerical_features=None)

if not result["success"]:
    print(result["message"])
    exit()

X_train = result["X_train"]
X_test = result["X_test"]
y_train = result["y_train"]
y_test = result["y_test"]

# Training the Simple Linear Regression model on the Training set
result = simple_linear_regression(X_train, X_test, y_train, y_test, result['scaler'])

if not result["success"]:
    print(result["message"])
    exit()

m = result["m"]
b = result["b"]
y = result["y"]
y_pred = result["y_pred"]
y_train_pred = result["y_train_pred"]
scaler = result["scaler"]

# Inverse transform the scaled data
X_train = scaler.inverse_transform(X_train)
X_test = scaler.inverse_transform(X_test)


# Visualising the Training set results
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, y, color='blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Visualising the Test set results
plt.scatter(X_test, y_test, color='red')
plt.plot(X_train, y, color='blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
