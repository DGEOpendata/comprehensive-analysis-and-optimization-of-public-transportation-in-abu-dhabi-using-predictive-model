python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load the dataset
url = 'https://example.com/Public_Transport_Utilization_Statistics_2022-2025.csv'
data = pd.read_csv(url)

# Preprocessing
# Convert date column to datetime
data['Date'] = pd.to_datetime(data['Date'])

# Feature engineering: Extract features like day of the week, month, etc.
data['DayOfWeek'] = data['Date'].dt.dayofweek
data['Month'] = data['Date'].dt.month

# Define features and target variable
drop_cols = ['Date', 'PassengerCount']
X = data.drop(columns=drop_cols)
y = data['PassengerCount']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate the model
predictions = model.predict(X_test)
rmse = mean_squared_error(y_test, predictions, squared=False)

# Print the RMSE
print(f'Root Mean Squared Error: {rmse}')

# Visualize the feature importance
importances = model.feature_importances_
feature_names = X.columns
plt.figure(figsize=(10, 6))
plt.title('Feature Importances')
plt.barh(range(len(importances)), importances, align='center')
plt.yticks(range(len(importances)), feature_names)
plt.xlabel('Relative Importance')
plt.show()
