import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

print("[INFO] Loading dataset.csv...")
# 1. Load the raw dataset
df = pd.read_csv('dataset.csv')
target_col = 'Disease' if 'Disease' in df.columns else 'prognosis'

print(f"[INFO] Target column identified as: '{target_col}'")
print(f"[INFO] Total rows in dataset: {len(df)}")

# 2. Extract all symptom columns dynamically (Symptom_1 to Symptom_17)
symptom_cols = [col for col in df.columns if col != target_col]

# Normalize symptom strings: strip whitespace and convert to lowercase for consistency
print("[INFO] Normalizing symptom text data...")
for col in symptom_cols:
    df[col] = df[col].astype(str).str.strip().str.lower()

# 3. Create a master global dictionary of all unique medical symptoms across the entire dataset
all_symptoms = set()
for col in symptom_cols:
    unique_vals = df[col].unique()
    for val in unique_vals:
        if pd.notna(val) and val != 'nan' and val != 'none' and val != '':
            all_symptoms.add(val)

all_symptoms = sorted(list(all_symptoms))
print(f"[SUCCESS] Total unique global symptoms indexed: {len(all_symptoms)}")

# 4. Transform tabular records into a Multi-Hot Binary Matrix (X)
print("[INFO] Constructing multi-hot binary feature matrix...")
X_rows = []
for idx, row in df.iterrows():
    # Collect all valid symptoms present in this particular disease record
    row_symptoms = {row[col] for col in symptom_cols if pd.notna(row[col])}
    # Create a 1/0 binary vector mapping against the master symptom list
    binary_vector = [1 if symptom in row_symptoms else 0 for symptom in all_symptoms]
    X_rows.append(binary_vector)

X = pd.DataFrame(X_rows, columns=all_symptoms)
y = df[target_col]

# 5. Train-Test Split (80% training, 20% testing for validation)
print("[INFO] Splitting dataset into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Initialize and Train the Random Forest Classifier
print("[INFO] Training Random Forest model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model accuracy on test data
accuracy = model.score(X_test, y_test)
print(f"[SUCCESS] Model training complete! Test Accuracy: {accuracy * 100:.2f}%")

# 7. Serialize and save artifacts for the web application
joblib.dump(model, 'model.pkl')
joblib.dump(all_symptoms, 'model_columns.pkl')
print("[SUCCESS] Saved 'model.pkl' and 'model_columns.pkl' successfully!")