import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

weights = np.array([800, 1000, 1200, 1400, 1600, 1800]).reshape(-1, 1)
fuel_efficiency = np.array([18, 16, 14, 12, 10, 8])

polynomial_features = PolynomialFeatures(degree=3)
weights_poly = polynomial_features.fit_transform(weights)

regression_model = LinearRegression()
regression_model.fit(weights_poly, fuel_efficiency)

weights_pred = np.linspace(700, 1900, 200).reshape(-1, 1)
weights_pred_poly = polynomial_features.transform(weights_pred)
predictions = regression_model.predict(weights_pred_poly)

plt.scatter(weights, fuel_efficiency, color='blue', label='Actual Data')
plt.plot(weights_pred, predictions, color='red', label='Polynomial Regression Line (degree 3)')
plt.xlabel('Weight (kg)')
plt.ylabel('Kilometers per liter')
plt.title('Polynomial Regression Degree 3 - Weight vs Fuel Efficiency')
plt.legend()
plt.grid(True)
plt.show()

prediction = regression_model.predict(polynomial_features.transform([[1500]]))
print(f"A car weighing 1500kg gets {prediction[0]:.2f} km per liter")
