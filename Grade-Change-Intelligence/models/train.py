import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
from xgboost import XGBClassifier

# -----------------------
# Load dataset
# -----------------------
df = pd.read_csv("data/historical_data.csv")

# Create transition identifier
df['transition_id'] = df['current_grade'] + "_" + df['target_grade']

# -----------------------
# Feature Engineering
# -----------------------
# Calculate rate of change (first difference) for each transition
for col in ['speed', 'stock_flow', 'steam', 'filler']:
    df[col+'_rate'] = df.groupby('transition_id')[col].diff().fillna(0)

# Target: forecast off-spec 30 seconds ahead
N = 30
df['target_off_spec'] = df.groupby('transition_id')['off_spec'].shift(-N).fillna(0).astype(int)

# -----------------------
# Train/Test Split by Grade-Transition
# -----------------------
tr_ids = df['transition_id'].unique()
train_transitions, test_transitions = train_test_split(tr_ids, test_size=0.2, random_state=42)

train_df = df[df['transition_id'].isin(train_transitions)]
test_df = df[df['transition_id'].isin(test_transitions)]

feature_cols = [
    "speed",
    "stock_flow",
    "steam",
    "filler",
    "speed_rate",
    "stock_flow_rate",
    "steam_rate",
    "filler_rate",
]

X_train = train_df[feature_cols]
y_train = train_df['target_off_spec']
X_test = test_df[feature_cols]
y_test = test_df['target_off_spec']

# -----------------------
# Train Model
# -----------------------
model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    random_state=42,
    eval_metric="logloss",
)

model.fit(X_train, y_train)

# -----------------------
# Predictions & Metrics
# -----------------------
pred = model.predict(X_test)

print("\nAccuracy :", accuracy_score(y_test, pred))
print("Precision:", precision_score(y_test, pred))
print("Recall   :", recall_score(y_test, pred))
print("F1 Score :", f1_score(y_test, pred))

print("\nClassification Report:\n")
print(classification_report(y_test, pred))

# -----------------------
# Save model
# -----------------------
joblib.dump(model, "models/model.pkl")

print("\nModel saved successfully!")