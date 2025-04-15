# -*- coding: utf-8 -*-
"""
Created on Tue Apr 15 18:00:50 2025

@author: Hp
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


np.random.seed(42)
random.seed(42)

num_orders = 10000
base_time = datetime(2024, 1, 1, 8, 0, 0)
order_times = [base_time + timedelta(minutes=random.randint(0, 60*24*60)) for _ in range(num_orders)]

zones = ['Zone_A', 'Zone_B', 'Zone_C', 'Zone_D', 'Zone_E']
restaurants = [random.choice(zones) for _ in range(num_orders)]
customers = [random.choice(zones) for _ in range(num_orders)]
traffic = [random.choice(['Low', 'Medium', 'High']) for _ in range(num_orders)]
weather = [random.choice(['Clear', 'Rainy', 'Cloudy']) for _ in range(num_orders)]
is_holiday = [random.choice([0, 1]) if t.weekday() in [5, 6] else 0 for t in order_times]
prep_times = [random.randint(5, 20) for _ in range(num_orders)]

zone_distance_map = {
    ('Zone_A', 'Zone_A'): 2, ('Zone_A', 'Zone_B'): 4, ('Zone_A', 'Zone_C'): 6,
    ('Zone_A', 'Zone_D'): 8, ('Zone_A', 'Zone_E'): 10, ('Zone_B', 'Zone_B'): 2,
    ('Zone_B', 'Zone_C'): 4, ('Zone_B', 'Zone_D'): 6, ('Zone_B', 'Zone_E'): 8,
    ('Zone_C', 'Zone_C'): 2, ('Zone_C', 'Zone_D'): 4, ('Zone_C', 'Zone_E'): 6,
    ('Zone_D', 'Zone_D'): 2, ('Zone_D', 'Zone_E'): 4, ('Zone_E', 'Zone_E'): 2
}

def get_distance(r, c):
    return zone_distance_map.get((r, c), zone_distance_map.get((c, r), 10))

distances = [get_distance(r, c) for r, c in zip(restaurants, customers)]

def simulate_delivery_time(dist, traffic, weather):
    base_speed = 30  # km/h
    if traffic == 'Medium': base_speed *= 0.75
    if traffic == 'High': base_speed *= 0.5
    if weather == 'Rainy': base_speed *= 0.85
    if weather == 'Cloudy': base_speed *= 0.95
    return int((dist / base_speed) * 60)

pickup_times = []
delivery_times = []

for i in range(num_orders):
    prep = prep_times[i]
    pickup_time = order_times[i] + timedelta(minutes=prep)
    travel_time = simulate_delivery_time(distances[i], traffic[i], weather[i])
    delivery_time = pickup_time + timedelta(minutes=travel_time)
    pickup_times.append(pickup_time)
    delivery_times.append(delivery_time)

df = pd.DataFrame({
    'order_id': range(1, num_orders + 1),
    'order_time': order_times,
    'restaurant_zone': restaurants,
    'customer_zone': customers,
    'traffic': traffic,
    'weather': weather,
    'is_holiday': is_holiday,
    'prep_time_min': prep_times,
    'delivery_distance_km': distances,
    'pickup_time': pickup_times,
    'delivery_time': delivery_times
})

df['delivery_duration_min'] = (df['delivery_time'] - df['order_time']).dt.total_seconds() / 60


features = [
    'restaurant_zone', 'customer_zone', 'traffic', 'weather',
    'is_holiday', 'prep_time_min', 'delivery_distance_km'
]
target = 'delivery_duration_min'

X = df[features]
y = df[target]

categorical_features = ['restaurant_zone', 'customer_zone', 'traffic', 'weather']
numeric_features = ['is_holiday', 'prep_time_min', 'delivery_distance_km']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('num', 'passthrough', numeric_features)
    ]
)

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n📊 Model Performance:")
print(f"Mean Absolute Error (MAE): {mae:.2f} mins")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f} mins")
print(f"R² Score: {r2:.5f}")


print("\n🚀 Real-Time Delivery Time Prediction")
def predict_delivery_time():
    restaurant_zone = input("Enter restaurant zone (Zone_A to Zone_E): ")
    customer_zone = input("Enter customer zone (Zone_A to Zone_E): ")
    traffic = input("Enter traffic condition (Low / Medium / High): ")
    weather = input("Enter weather (Clear / Cloudy / Rainy): ")
    is_holiday = int(input("Is it a holiday? (0 = No, 1 = Yes): "))
    prep_time_min = int(input("Enter restaurant prep time in minutes: "))
    delivery_distance_km = float(input("Enter delivery distance in km: "))

    input_data = pd.DataFrame([{
        'restaurant_zone': restaurant_zone,
        'customer_zone': customer_zone,
        'traffic': traffic,
        'weather': weather,
        'is_holiday': is_holiday,
        'prep_time_min': prep_time_min,
        'delivery_distance_km': delivery_distance_km
    }])

    prediction = model.predict(input_data)[0]
    print(f"\n⏱️ Predicted Delivery Duration: {prediction:.2f} minutes\n")

while True:
    predict_delivery_time()
    again = input("Do you want to predict another? (yes/no): ").strip().lower()
    if again != 'yes':
        print("Exiting prediction mode.")
        break
