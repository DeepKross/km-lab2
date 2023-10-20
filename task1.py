import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf


# Оголошуємо функцію та її похідну
def f(x):
    return erf(x)


def df(x):
    return (2 / np.sqrt(np.pi)) * np.exp(-x ** 2)


# Визначимо чисельне диференціювання за формулою центральної різниці
def central_difference(f, x, h):
    return (f(x + h) - f(x - h)) / (2 * h)


# Обчислюємо похибку для різних значень кроку h
h_values = np.logspace(-8, -0.5, 100)  # створюємо масив значень кроку h
xi = 0.5  # обране значення x для обчислення похідної
errors = []

for h in h_values:
    numerical_derivative = central_difference(f, xi, h)
    actual_derivative = df(xi)
    error = abs(numerical_derivative - actual_derivative)
    errors.append(error)

# Знаходимо крок при якому похибка мінімальна
min_error_index = np.argmin(errors)
optimal_h = h_values[min_error_index]

# Візуалізуємо результати
plt.figure(figsize=(10, 6))
plt.loglog(h_values, errors, label='Похибка чисельного диференціювання')
plt.axvline(optimal_h, color='red', linestyle='--', label=f"Оптимальний крок h={optimal_h:.1e}")
plt.xlabel("Крок h")
plt.ylabel("Похибка")
plt.title("Залежність похибки від кроку h")
plt.legend()
plt.grid(True)
plt.show()

print(f"Оптимальний крок h={optimal_h}")
