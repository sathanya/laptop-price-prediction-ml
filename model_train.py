import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import joblib

# Load dataset
df = pd.read_csv("D:/ML demo/laptopPrice.csv")

# Convert RAM, SSD, HDD to int
df['ram_gb'] = df['ram_gb'].str.replace(' GB', '').astype(int)
df['ssd'] = df['ssd'].str.replace(' GB', '').astype(int)
df['hdd'] = df['hdd'].str.replace(' GB', '').astype(int)

# Drop unnecessary columns
df.drop(['warranty', 'rating', 'weight', 'graphic_card_gb',
         'Touchscreen', 'msoffice', 'Number of Ratings', 'Number of Reviews'], axis=1, inplace=True)

# Fill missing values
df.fillna('Not Available', inplace=True)

# Encode categorical columns
encoders = {}
for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le  # Save encoder for each categorical column

# Features and target
features = ['brand', 'processor_brand', 'processor_name', 'processor_gnrtn',
            'ram_gb', 'ram_type', 'ssd', 'hdd']
X = df[features]
y = df['Price']

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Save model, features, and encoders
joblib.dump(model, 'laptop_price_model.pkl')
joblib.dump(features, 'features.pkl')
joblib.dump(encoders, 'encoders.pkl')
