import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle
import warnings
warnings.filterwarnings('ignore')

print("=" * 50)
print("Training Weather Prediction Model")
print("=" * 50)

# Load data
try:
    df = pd.read_csv('seattle-weather.csv')
    print("Data loaded successfully!")
    print(f"Dataset shape: {df.shape}")
except FileNotFoundError:
    print("Error: seattle-weather.csv not found!")
    print("Please ensure the file is in the same directory.")
    exit(1)

# Feature engineering
df['date'] = pd.to_datetime(df['date'])
df['is_sunny'] = (df['weather'] == 'sun').astype(int)
df['day'] = df['date'].dt.day
df['month'] = df['date'].dt.month
df['year'] = df['date'].dt.year
df['day_of_week'] = df['date'].dt.dayofweek

print("\nFeature engineering completed!")

# Select features
features = ['precipitation', 'temp_max', 'temp_min', 'wind', 'day', 'month', 'year', 'day_of_week']
X = df[features]
y = df['is_sunny']

print(f"\nFeatures: {features}")
print(f"Target: is_sunny (1 = Sunny, 0 = Not Sunny)")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# Train model
print("\nTraining Random Forest Classifier...")
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nModel Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Not Sunny', 'Sunny']))

# Feature importance
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nFeature Importance:")
for _, row in feature_importance.iterrows():
    print(f"  {row['feature']}: {row['importance']:.4f}")

# Save model
with open('weather_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("\n" + "=" * 50)
print("Model saved as 'weather_model.pkl'")
print("=" * 50)
