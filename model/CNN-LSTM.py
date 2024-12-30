# %%
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt

# ====================== Data Loading ======================
train_data = pd.read_csv('/Users/jon/Codes/MyProjects/PythonProjects/arrhythmia-classification/data/train.csv')
test_data = pd.read_csv('/Users/jon/Codes/MyProjects/PythonProjects/arrhythmia-classification/data/test.csv')

# Features and Labels
X_train = train_data.drop(columns=['label'])
y_train = train_data['label']
X_test = test_data.drop(columns=['label'])
y_test = test_data['label']

# Label Encoding
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)

# Data Standardization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Handle NaN and Inf values
X_train = np.nan_to_num(X_train)
X_test = np.nan_to_num(X_test)

# Compute Class Weights
class_counts = np.bincount(y_train)
class_weights = {i: max(class_counts) / class_counts[i] for i in range(len(class_counts))}
class_weights = {key: min(value, 10) for key, value in class_weights.items()}  # Cap the weights
print(f"Class Weights: {class_weights}")

# ====================== Model Construction ======================
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(X_train.shape[1],)))
model.add(BatchNormalization())  # Regularization
model.add(Dropout(0.3))

model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Dense(len(label_encoder.classes_), activation='softmax'))

# ====================== Model Compilation ======================
optimizer = Adam(learning_rate=0.0001)  # Adjust learning rate
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# ====================== Callbacks ======================
early_stopping = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6, verbose=1)

# ====================== Model Training ======================
history = model.fit(
    X_train, y_train,
    epochs=700,
    batch_size=64,
    validation_data=(X_test, y_test),
    class_weight=class_weights,
    callbacks=[early_stopping, reduce_lr]
)

# ====================== Model Evaluation ======================
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"\nTest Accuracy: {test_accuracy * 100:.2f}%")

# ====================== Classification Report ======================
y_pred = model.predict(X_test).argmax(axis=1)

# Generate dynamic classification report for existing labels
existing_labels = np.unique(y_pred)
print("\nClassification Report (dynamic labels):")
print(classification_report(
    y_test,
    y_pred,
    labels=existing_labels,
    target_names=[str(label_encoder.classes_[i]) for i in existing_labels]  # Ensure strings
))

# Generate classification report for all classes
print("\nClassification Report (all classes):")
print(classification_report(
    y_test,
    y_pred,
    labels=range(len(label_encoder.classes_)),  # Include all classes
    target_names=[str(cls) for cls in label_encoder.classes_],  # Ensure strings
    zero_division=0  # Avoid division by zero
))

# ====================== Additional Metrics ======================
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

print(f"\nPrecision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(conf_matrix)

# ====================== Training Curve Plot ======================
def plot_training_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))

    # Accuracy curve
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, label='Training Accuracy', marker='o')
    plt.plot(epochs, val_acc, label='Validation Accuracy', marker='o')
    plt.title('Accuracy Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss curve
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, label='Training Loss', marker='o')
    plt.plot(epochs, val_loss, label='Validation Loss', marker='o')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Call the function to plot training history
plot_training_history(history)