# %%
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt

# 加载数据
train_data = pd.read_csv('/Users/jon/Codes/MyProjects/PythonProjects/arrhythmia-classification/data/train.csv')
test_data = pd.read_csv('/Users/jon/Codes/MyProjects/PythonProjects/arrhythmia-classification/data/test.csv')

# 特征和标签
X_train = train_data.drop(columns=['label'])
y_train = train_data['label']
X_test = test_data.drop(columns=['label'])
y_test = test_data['label']

# 合并训练集和测试集的标签
all_labels = np.concatenate([y_train, y_test])

# 使用 fit() 训练标签编码器
label_encoder = LabelEncoder()
label_encoder.fit(all_labels)

# 转换训练集和测试集的标签
y_train = label_encoder.transform(y_train)
y_test = label_encoder.transform(y_test)

# 标准化特征
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 计算类别权重来处理不平衡数据
class_counts = np.bincount(y_train)
class_weights = {i: max(class_counts) / class_counts[i] for i in range(len(class_counts))}
print(f"Class Weights: {class_weights}")

# 构建CNN模型（增加复杂度）
model = Sequential()

# 输入层及第一个隐藏层
model.add(Dense(512, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dropout(0.3))

# 第二个隐藏层
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))

# 第三个隐藏层
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))

# 输出层
model.add(Dense(len(label_encoder.classes_), activation='softmax'))

# 编译模型（使用学习率调度器）
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 设置回调函数
early_stopping = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6, verbose=1)

# 训练模型
history = model.fit(
    X_train, y_train,
    epochs=700,
    batch_size=64,
    validation_data=(X_test, y_test),
    class_weight=class_weights,
    callbacks=[early_stopping, reduce_lr]
)

# 模型评估
print("Evaluating model on test data:")
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# 分类报告
y_pred = model.predict(X_test).argmax(axis=1)
target_names = [str(cls) for cls in label_encoder.classes_]  # 转换为字符串类型

print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=target_names))

# 绘制训练曲线
def plot_training_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))

    # 绘制准确率曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, label='Training Accuracy', marker='o')
    plt.plot(epochs, val_acc, label='Validation Accuracy', marker='o')
    plt.title('Accuracy Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # 绘制损失曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, label='Training Loss', marker='o')
    plt.plot(epochs, val_loss, label='Validation Loss', marker='o')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

plot_training_history(history)