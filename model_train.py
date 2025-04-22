import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import joblib

# Load dataset
df = pd.read_csv("D:/ML demo/laptopPrice.csv")

# Clean and preprocess
df['ram_gb'] = df['ram_gb'].str.replace(' GB', '').astype(int)
df['ssd'] = df['ssd'].str.replace(' GB', '').astype(int)
df['hdd'] = df['hdd'].str.replace(' GB', '').astype(int)

# Drop unnecessary columns
df.drop(['warranty', 'rating', 'weight', 'graphic_card_gb',
         'Touchscreen', 'msoffice', 'Number of Ratings', 'Number of Reviews'], axis=1, inplace=True)

# Handle missing values
df.fillna('Not Available', inplace=True)

# Encode categorical columns
le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])

# Select only the most essential features
X = df[['brand', 'processor_brand', 'processor_name', 'processor_gnrtn',
        'ram_gb', 'ram_type', 'ssd', 'hdd']]
y = df['Price']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Save model and features
joblib.dump(model, 'laptop_price_model.pkl')
joblib.dump(X.columns.tolist(), 'features.pkl')
