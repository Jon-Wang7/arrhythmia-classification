# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from collections import Counter

# ====================== Data Loading ======================
train_data = pd.read_csv('/Users/jon/Codes/MyProjects/PythonProjects/arrhythmia-classification/data/train.csv')
test_data = pd.read_csv('/Users/jon/Codes/MyProjects/PythonProjects/arrhythmia-classification/data/test.csv')

# Features and labels
X_train = train_data.drop(columns=['label']).values
y_train = train_data['label'].values
X_test = test_data.drop(columns=['label']).values
y_test = test_data['label'].values

# Label encoding
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)

# Data standardization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Data cleaning: Replace NaN and Inf values
X_train = np.nan_to_num(X_train)
X_test = np.nan_to_num(X_test)

# Check class distribution before filtering
print("\nClass distribution before filtering:")
print(Counter(y_train))

# Remove classes with fewer than k_neighbors + 1 samples
k_neighbors = 5
min_samples_required = k_neighbors + 1
label_counts = Counter(y_train)
valid_labels = [label for label, count in label_counts.items() if count >= min_samples_required]

X_train = X_train[np.isin(y_train, valid_labels)]
y_train = y_train[np.isin(y_train, valid_labels)]

# Check class distribution after filtering
print("\nClass distribution after filtering:")
print(Counter(y_train))

# Use SMOTE to balance class distribution
smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Check class distribution after SMOTE
print("\nClass distribution after SMOTE:")
print(Counter(y_train))

# ====================== Simulated Epoch Training ======================
n_estimators_range = range(10, 310, 10)  # Simulate 30 "epochs"
train_accuracies = []
test_accuracies = []

for n_estimators in n_estimators_range:
    rf_model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=None,
        random_state=42,
        class_weight='balanced'
    )
    rf_model.fit(X_train, y_train)

    # Training and validation accuracy
    train_accuracy = accuracy_score(y_train, rf_model.predict(X_train))
    test_accuracy = accuracy_score(y_test, rf_model.predict(X_test))

    train_accuracies.append(train_accuracy)
    test_accuracies.append(test_accuracy)

# ====================== Plotting Training and Validation Curves ======================
plt.figure(figsize=(12, 5))

# Accuracy curve
plt.subplot(1, 2, 1)
plt.plot(n_estimators_range, train_accuracies, label='Training Accuracy', marker='o')
plt.plot(n_estimators_range, test_accuracies, label='Validation Accuracy', marker='o')
plt.title('Accuracy Over Simulated Epochs')
plt.xlabel('Simulated Epochs (Number of Estimators)')
plt.ylabel('Accuracy')
plt.legend()

# Pseudo-loss curve (1 - Accuracy)
pseudo_loss_train = [1 - acc for acc in train_accuracies]
pseudo_loss_test = [1 - acc for acc in test_accuracies]

plt.subplot(1, 2, 2)
plt.plot(n_estimators_range, pseudo_loss_train, label='Training Loss', marker='o')
plt.plot(n_estimators_range, pseudo_loss_test, label='Validation Loss', marker='o')
plt.title('Pseudo-Loss Over Simulated Epochs')
plt.xlabel('Simulated Epochs (Number of Estimators)')
plt.ylabel('Pseudo-Loss (1 - Accuracy)')
plt.legend()

plt.tight_layout()
plt.show()

# ====================== Final Model and Evaluation ======================
# Train with final number of estimators
final_rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    random_state=42,
    class_weight='balanced'
)
final_rf_model.fit(X_train, y_train)

# Model evaluation
y_pred = final_rf_model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy: {accuracy * 100:.2f}%")

# Classification report
print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    labels=range(len(label_encoder.classes_)),  # Include all classes
    target_names=[str(cls) for cls in label_encoder.classes_],  # Ensure all class names are included
    zero_division=0  # Handle classes with no predictions gracefully
))

# Additional metrics
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

print(f"\nPrecision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(conf_matrix)