import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# training data
x_train = np.linspace(-3, 3, 20)
y_train = x_train**2 + np.random.normal(0, 1.5, size=len(x_train))

# test data
x_test = np.linspace(-3, 3, 100)
y_test = x_test**2 + np.random.normal(0, 1.5, size=len(x_test))

def mse(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)

degrees = [1, 2, 10]

for degree in degrees:
    coeffs = np.polyfit(x_train, y_train, degree)
    poly = np.poly1d(coeffs)

    train_pred = poly(x_train)
    test_pred = poly(x_test)

    train_error = mse(train_pred, y_train)
    test_error = mse(test_pred, y_test)

    print(f"Degree {degree}")
    print("  Train error:", train_error)
    print("  Test error :", test_error)
    
x_plot = np.linspace(-3, 3, 400)

plt.scatter(x_train, y_train, label="train data")

for degree in degrees:
    coeffs = np.polyfit(x_train, y_train, degree)
    poly = np.poly1d(coeffs)
    plt.plot(x_plot, poly(x_plot), label=f"degree {degree}")

plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.title("Underfitting vs Overfitting")
plt.show()