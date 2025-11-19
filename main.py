# | Altitude(m)  | Temperature(°C)  |
# | ------------ | ---------------- |
# | 0            | 30               |
# | 500          | 27               |
# | 1000         | 24               |
# | 1500         | 21               |
# | 2000         | 18               |

# Questions:
# 1. Fit the model
# 2. Find the slope and intercept
# 3. Predict temperature at 1200m
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# 1. Read CSV
data = pd.read_csv('temp.csv')

# 2. Separate features and target
Altitude = data['Altitude'].values.reshape(-1, 1)
Temperature = data['Temperature'].values

# 3. Fit the model
model = LinearRegression()
model.fit(Altitude, Temperature)

# 4. Print slope and intercept
print("Slope (coefficient):", model.coef_[0])
print("Intercept:", model.intercept_)

# 5. Predict temperature at 1200m
predicted_temp = model.predict([[1200]])
print("Predicted temperature at 1200m:", predicted_temp[0])

# 6. Plot
plt.scatter(Altitude, Temperature, color='blue', label='Data Points')
plt.plot(Altitude, model.predict(Altitude),
         color='red', label='Regression Line')

# Mark prediction
plt.scatter(1200, predicted_temp, color='green',
            s=100, label='Predicted @ 1200m')

plt.xlabel('Altitude (m)')
plt.ylabel('Temperature (°C)')
plt.title('Altitude vs Temperature with Linear Regression')
plt.legend()
plt.grid(True)
plt.show()
