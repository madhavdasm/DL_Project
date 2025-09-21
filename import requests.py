import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

df = pd.read_csv('aircraft_accidents.csv')   # Replace with actual filename

# Convert date columns
df['Incident_Date'] = pd.to_datetime(df['Incident_Date'], errors='coerce')
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# Drop rows with missing critical info
df.dropna(subset=['Incident_Date', 'Aircaft_Model', 'Aircaft_Operator', 'Incident_Category'], inplace=True)

# Fill missing numeric values
numeric_cols = ['Fatalities', 'Onboard_Total', 'Ground_Casualties', 'Collision_Casualties']
df[numeric_cols] = df[numeric_cols].fillna(0)

# Encode categorical features
cat_cols = ['Aircaft_Model', 'Aircaft_Operator', 'Aircaft_Nature', 'Incident_Category',
            'Incident_Cause(es)', 'Aircaft_Damage_Type', 'Aircraft_Phase']
encoder = LabelEncoder()
for col in cat_cols:
    df[col] = encoder.fit_transform(df[col].astype(str))

# Normalize numeric features
scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
