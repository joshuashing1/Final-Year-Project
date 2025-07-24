import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Define Nelson-Siegel
def nelson_siegel(tau, beta1, beta2, beta3, lambd):
    term1 = (1 - np.exp(-lambd * tau)) / (lambd * tau)
    term2 = term1 - np.exp(-lambd * tau)
    return beta1 + beta2 * term1 + beta3 * term2

# Define Svensson
def svensson(tau, beta1, beta2, beta3, beta4, lambd1, lambd2):
    term1 = (1 - np.exp(-lambd1 * tau)) / (lambd1 * tau)
    term2 = term1 - np.exp(-lambd1 * tau)
    term3 = (1 - np.exp(-lambd2 * tau)) / (lambd2 * tau) - np.exp(-lambd2 * tau)
    return beta1 + beta2 * term1 + beta3 * term2 + beta4 * term3

# Example data (replace with your observed data)
# tau_data = np.array([0.25, 0.5, 1, 2, 5, 10, 20, 30])  # maturities in years
# y_data = np.array([5.1, 5.15, 5.0, 4.7, 4.5, 4.4, 4.6, 4.55])  # yields

# For demonstration, generate some synthetic data
np.random.seed(1)
tau_data = np.array([0.25, 0.5, 1, 2, 5, 10, 20, 30])
y_data = nelson_siegel(tau_data, 5.2, -1.1, 1.2, 0.7) + 0.05 * np.random.randn(len(tau_data))

# Choose model: 'nelson_siegel' or 'svensson'
parameterization = 'nelson_siegel'

if parameterization == 'nelson_siegel':
    # Initial guess for [beta1, beta2, beta3, lambd]
    p0 = [4.0, -1.0, 1.0, 0.5]
    params, _ = curve_fit(nelson_siegel, tau_data, y_data, p0=p0, maxfev=10000)
    y_fit = nelson_siegel(tau_data, *params)
    print(f"Nelson-Siegel params: {params}")

elif parameterization == 'svensson':
    # Initial guess for [beta1, beta2, beta3, beta4, lambd1, lambd2]
    p0 = [4.0, -1.0, 1.0, 0.5, 0.5, 0.2]
    params, _ = curve_fit(svensson, tau_data, y_data, p0=p0, maxfev=10000)
    y_fit = svensson(tau_data, *params)
    print(f"Svensson params: {params}")

# Plot the results
plt.figure(figsize=(8,4))
plt.plot(tau_data, y_data, 'o', label='Observed yields')
plt.plot(tau_data, y_fit, '-', label=f'Fitted {parameterization.replace("_", " ").title()}')
plt.xlabel('Maturity (years)')
plt.ylabel('Yield (%)')
plt.legend()
plt.grid(True)
plt.show()
