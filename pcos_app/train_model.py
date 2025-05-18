import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Load the synthetic dataset
df = pd.read_csv("synthetic_pcos_dataset.csv")

# Create a synthetic target variable based on certain conditions
df['PCOS'] = np.where(
    (df['Testosterone_Level_ng_dl'] > 80) |
    (df['Cycle_Regularity'] == 'Irregular') |
    (df['Hirsutism_Score'] > 5) |
    (df['Family_History_PCOS'] == 'Yes'),
    1, 0
)

# Separate features and target
X = df.drop(['PCOS'], axis=1)
y = df['PCOS']

# Encode categorical variables
categorical_cols = X.select_dtypes(include=['object']).columns
le_dict = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    le_dict[col] = le

# Save label encoders for future use
joblib.dump(le_dict, 'label_encoders.pkl')

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save the scaler
joblib.dump(scaler, 'scaler.pkl')

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Initialize base estimator for RFECV
base_estimator = RandomForestClassifier(random_state=42)

# Recursive Feature Elimination with Cross-Validation
rfecv = RFECV(
    estimator=base_estimator,
    step=1,
    cv=StratifiedKFold(5),
    scoring='accuracy',
    n_jobs=-1
)
rfecv.fit(X_train, y_train)

# Transform training and testing sets
X_train_rfe = rfecv.transform(X_train)
X_test_rfe = rfecv.transform(X_test)

# Save the RFECV selector
joblib.dump(rfecv, 'rfecv_selector.pkl')

# Define base learners for stacking
base_learners = [
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('svc', SVC(probability=True, random_state=42)),
    ('nb', GaussianNB()),
    ('ada', AdaBoostClassifier(n_estimators=100, random_state=42))
]

# Define meta-learner
meta_learner = LogisticRegression()

# Initialize Stacking Classifier
stacking_clf = StackingClassifier(
    estimators=base_learners,
    final_estimator=meta_learner,
    cv=5,
    n_jobs=-1
)

# Train the stacking classifier
stacking_clf.fit(X_train_rfe, y_train)

# Save the trained model
joblib.dump(stacking_clf, 'pcos_model.pkl')

# Predict on the test set
y_pred = stacking_clf.predict(X_test_rfe)
y_proba = stacking_clf.predict_proba(X_test_rfe)[:, 1]

# Evaluation Metrics
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print(f"ROC AUC Score: {roc_auc_score(y_test, y_proba):.4f}")