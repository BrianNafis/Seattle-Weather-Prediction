import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('seattle-weather.csv')

# Feature engineering
df['date'] = pd.to_datetime(df['date'])
df['is_sunny'] = (df['weather'] == 'sun').astype(int)
df['day'] = df['date'].dt.day
df['month'] = df['date'].dt.month
df['year'] = df['date'].dt.year
df['day_of_week'] = df['date'].dt.dayofweek

# Select features
features = ['precipitation', 'temp_max', 'temp_min', 'wind', 'day', 'month', 'year', 'day_of_week']
X = df[features]
y = df['is_sunny']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save model
with open('weather_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("\nModel saved as 'weather_model.pkl'")
