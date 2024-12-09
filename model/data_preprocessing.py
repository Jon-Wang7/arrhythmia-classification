import os
import wfdb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

class ECGDataLoader:
    def __init__(self, data_dir, label_file, max_records=5000):
        self.data_dir = data_dir
        self.label_file = label_file
        self.max_records = max_records
        self.ecg_data = []
        self.labels = []
        self.labels_code = []
        self.record_names = []

    def load_labels(self):
        """加载 ConditionNames_SNOMED-CT.csv 中的标签信息"""
        label_data = pd.read_csv(self.label_file)
        # 创建标签映射字典
        label_map = dict(zip(label_data['Snomed_CT'], label_data['Acronym Name']))
        return label_map

    def extract_features(self, signal):
        """从 ECG 信号中提取特征（这里是简单的时域特征）"""
        # 简单的特征提取：均值、标准差、最大值和最小值
        mean_signal = np.mean(signal, axis=0)
        std_signal = np.std(signal, axis=0)
        max_signal = np.max(signal, axis=0)
        min_signal = np.min(signal, axis=0)

        # 返回特征
        return np.concatenate([mean_signal, std_signal, max_signal, min_signal])

    def load_and_preprocess_data(self):
        """加载并处理数据"""
        label_map = self.load_labels()
        count = 0

        for root, _, files in os.walk(self.data_dir):
            for file in files:
                if file.endswith('.mat'):
                    try:
                        # 构建记录路径（不包含扩展名）
                        record_name = file[:-4]
                        record_path = os.path.join(root, record_name)

                        # 读取ECG信号数据
                        record = wfdb.rdrecord(record_path)
                        ecg_signal = record.p_signal

                        # 假设 Dx 字段包含病症信息，映射到标签
                        dx_field = record.comments[2]  # 假设 Dx 信息位于评论的第 3 项
                        if dx_field:
                            # 清理 Dx 字段，去掉多余的部分（如 'Dx: '）
                            dx_codes = dx_field.split(":")[1].strip()  # 处理 Dx 信息
                            label_code = int(dx_codes.split(",")[0].strip())
                            # 获取对应标签（假设是病症的首个诊断）
                            label = label_map.get(label_code, 'Unknown')  # 如果没有找到标签则返回 'Unknown'
                        else:
                            label = 'Unknown'
                            label_code = 'Unknown'

                        # 提取特征并添加标签
                        features = self.extract_features(ecg_signal)
                        self.ecg_data.append(features)
                        self.labels.append(label)
                        self.labels_code.append(label_code)
                        self.record_names.append(record_name)

                        count += 1
                        if count >= self.max_records:
                            # 返回为 NumPy 数组
                            return np.array(self.ecg_data), np.array(self.labels), np.array(self.labels_code), self.record_names
                    except Exception as e:
                        print(f"Error loading {file}: {e}")
                        continue

        # Ensure the return is NumPy arrays
        return np.array(self.ecg_data), np.array(self.labels),np.array(self.labels_code), self.record_names

    def preprocess_labels(self):
        """将标签转化为整数编码或OneHot编码"""
        le = LabelEncoder()
        y_encoded = le.fit_transform(self.labels)

        # OneHot编码
        ohe = OneHotEncoder(sparse_output=False)
        y_one_hot = ohe.fit_transform(y_encoded.reshape(-1, 1))

        return y_encoded, y_one_hot, le

    def get_data_split(self):
        """划分训练集和测试集"""
        X_train, X_test, y_train, y_test = train_test_split(self.ecg_data, self.labels, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test

    def save_to_csv(self, output_file='../data/preprocessed_data.csv'):
        """将预处理数据保存为 CSV 文件"""
        # 将特征和标签合并到一个 DataFrame 中
        df = pd.DataFrame(self.ecg_data)
        df['label'] = self.labels  # 添加标签列
        df['label_code'] = self.labels_code  # 添加 label_code 列

        # 保存为 CSV 文件
        df.to_csv(output_file, index=False)
        print(f"Data saved to {output_file}")


def main():
    # 数据目录和标签文件路径
    data_dir = "../data/WFDBRecords"  # 数据文件路径
    label_file = "../data/ConditionNames_SNOMED-CT.csv"  # 标签文件路径

    # 创建 ECGDataLoader 实例
    ecg_loader = ECGDataLoader(data_dir, label_file)

    # 加载并处理数据，接收四个返回值
    ecg_data, labels, labels_code, record_names = ecg_loader.load_and_preprocess_data()

    # 打印数据的基本信息
    print(f"Loaded {len(ecg_data)} ECG records.")
    print(f"Sample features: {ecg_data[0]}")
    print(f"Sample label: {labels[0]}")
    print(f"Sample label code: {labels_code[0]}")  # 打印label_code

    # 标签处理：编码标签
    y_encoded, y_one_hot, label_encoder = ecg_loader.preprocess_labels()

    # 数据划分
    X_train, X_test, y_train, y_test = ecg_loader.get_data_split()

    # 将数据保存为 CSV 文件
    ecg_loader.save_to_csv('../data/preprocessed_data.csv')


if __name__ == "__main__":
    main()