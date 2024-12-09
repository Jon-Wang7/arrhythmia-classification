import os
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from data_preprocessing import load_ecg_data, preprocess_labels, split_data
from CNN import build_cnn_model


def train_and_evaluate():
    # 路径设置
    data_dir = "data/WFDBRecords"
    label_file = os.path.join(data_dir, "ConditionNames_SNOMED-CT.csv")

    # 数据加载与预处理
    print("Loading and preprocessing data...")
    ecg_data, record_labels, record_files = load_ecg_data(data_dir)
    encoded_labels, one_hot_labels, label_encoder = preprocess_labels(label_file, record_labels)
    X_train, X_test, y_train, y_test = split_data(ecg_data, one_hot_labels)

    # 模型构建
    print("Building model...")
    input_shape = X_train.shape[1:]
    num_classes = y_train.shape[1]
    model = build_cnn_model(input_shape, num_classes)

    # 回调函数
    checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_loss', mode='min')
    early_stop = EarlyStopping(monitor='val_loss', patience=5, mode='min')

    # 模型训练
    print("Training model...")
    history = model.fit(X_train, y_train,
                        validation_data=(X_test, y_test),
                        epochs=20, batch_size=32,
                        callbacks=[checkpoint, early_stop])

    # 模型评估
    print("Evaluating model...")
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

    # 标签映射测试
    sample_predictions = model.predict(X_test[:5])
    predicted_classes = label_encoder.inverse_transform(sample_predictions.argmax(axis=1))
    print(f"Predicted classes: {predicted_classes}")


if __name__ == "__main__":
    train_and_evaluate()