import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils import class_weight
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import numpy as np
import re

# --- Loading the Dataset ---
file_path = 'aircraft_accidents.csv'
df = pd.read_csv(file_path)

# --- Improved Data Cleaning & Preprocessing ---
def parse_occupant_string(value, part='fatalities'):
    if not isinstance(value, str):
        return value if pd.notna(value) else 0
    numbers = re.findall(r'\d+', value)
    if len(numbers) >= 2:
        return float(numbers[0] if part == 'fatalities' else numbers[1])
    elif len(numbers) == 1:
        return float(numbers[0])
    return 0

df['Fatalities'] = df['Fatalities'].apply(lambda x: parse_occupant_string(x, part='fatalities'))
df['Onboard_Total'] = df['Onboard_Total'].apply(lambda x: parse_occupant_string(x, part='total'))
df['Ground_Casualties'] = pd.to_numeric(df['Ground_Casualties'], errors='coerce').fillna(0)

df['Incident_Date'] = pd.to_datetime(df['Incident_Date'], errors='coerce', format='mixed')
df['Aircaft_First_Flight'] = pd.to_datetime(df['Aircaft_First_Flight'], errors='coerce', format='mixed')
df.dropna(subset=['Incident_Date', 'Aircaft_Model', 'Aircaft_Operator', 'Incident_Category'], inplace=True)


# --- Step 1: Define a Target Variable (Y) ---
def assign_severity(fatalities):
    if fatalities == 0: return 0
    elif 1 <= fatalities <= 10: return 1
    else: return 2

df['Severity'] = df['Fatalities'].apply(assign_severity)
y_labels = df['Severity'].values
y = to_categorical(y_labels, num_classes=3)


# --- Step 2: Advanced Feature Engineering with One-Hot Encoding ---
print("Performing advanced feature engineering...")
df['Incident_Year'] = df['Incident_Date'].dt.year
df['Incident_Month'] = df['Incident_Date'].dt.month
df['Incident_DayOfWeek'] = df['Incident_Date'].dt.dayofweek

cat_cols = [
    'Aircaft_Nature',
    'Incident_Category',
    'Aircaft_Damage_Type',
    'Aircraft_Phase'
]

# **ACCURACY BOOST: Switched from LabelEncoder to One-Hot Encoding**
# This is a much better way to represent categorical data for a neural network.
print("Applying One-Hot Encoding to categorical features...")
df_encoded = pd.get_dummies(df, columns=cat_cols, prefix=cat_cols)

df['Aircraft_Age_Days'] = (df['Incident_Date'] - df['Aircaft_First_Flight']).dt.days
mean_age = df['Aircraft_Age_Days'].mean()
df_encoded['Aircraft_Age_Days'] = df['Aircraft_Age_Days'].fillna(mean_age)

numeric_features_for_model = ['Onboard_Total', 'Ground_Casualties', 'Aircraft_Age_Days', 'Incident_Year', 'Incident_Month', 'Incident_DayOfWeek']
scaler = StandardScaler()
df_encoded[numeric_features_for_model] = scaler.fit_transform(df_encoded[numeric_features_for_model])

# Create the final feature list by combining numeric and the new one-hot encoded columns
one_hot_cols = [col for col in df_encoded.columns if any(cat_col in col for cat_col in cat_cols)]
features = numeric_features_for_model + one_hot_cols
X = df_encoded[features]
print(f"Total number of features after One-Hot Encoding: {X.shape[1]}")


# --- Step 3: Split the Data ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# --- Handle Class Imbalance ---
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_labels), y=y_labels)
class_weights_dict = dict(enumerate(class_weights))


# --- Step 4: Build a Deeper & More Powerful MLP Model ---
print("--- Building a Deeper MLP Model ---")
model = Sequential([
    Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.4),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(3, activation='softmax')
])

optimizer = Adam(learning_rate=0.0005)
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()


# --- Use EarlyStopping ---
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# --- Step 5: Train the Model ---
print("\n--- Training the Deeper Model ---")
history = model.fit(X_train, y_train,
                    validation_data=(X_test, y_test),
                    epochs=100,
                    batch_size=32,
                    class_weight=class_weights_dict,
                    callbacks=[early_stopping],
                    verbose=1)


# --- Step 6: Evaluate and Visualize ---
print("\n--- Evaluating Final Model Performance ---")
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {accuracy*100:.2f}%")
print(f"Test Loss: {loss:.4f}")

# Plotting code remains the same...
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()

plt.tight_layout()
plt.show()

