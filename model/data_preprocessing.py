import os
import wfdb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


class ECGDataLoader:
    def __init__(self, data_dir, label_file, max_records=5000):
        self.data_dir = data_dir
        self.label_file = label_file
        self.max_records = max_records
        self.ecg_data = []
        self.labels = []
        self.labels_code = []
        self.ages = []
        self.genders = []
        self.record_names = []
        self.feature_columns = []  # 用于存储每列的含义

    def load_labels(self):
        """加载 ConditionNames_SNOMED-CT.csv 中的标签信息"""
        label_data = pd.read_csv(self.label_file)
        # 创建标签映射字典
        label_map = dict(zip(label_data['Snomed_CT'], label_data['Acronym Name']))
        return label_map

    def extract_features(self, signal):
        """从 ECG 信号中提取特征（包含详细列名）"""
        lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

        # 计算统计特征
        mean_signal = np.mean(signal, axis=0)
        std_signal = np.std(signal, axis=0)
        max_signal = np.max(signal, axis=0)
        min_signal = np.min(signal, axis=0)

        # 更新列名
        self.feature_columns = []
        for lead in lead_names:
            self.feature_columns.extend([
                f"{lead}_mean",
                f"{lead}_std",
                f"{lead}_max",
                f"{lead}_min"
            ])

        # 返回特征
        return np.concatenate([mean_signal, std_signal, max_signal, min_signal])

    def parse_metadata(self, comments):
        """解析元数据中的性别和年龄信息"""
        age = None
        gender = None
        for comment in comments:
            if comment.startswith("Age:"):
                try:
                    age = int(comment.split(":")[1].strip())
                except ValueError:
                    age = None
            if comment.startswith("Sex:"):
                gender = comment.split(":")[1].strip()
        return age, gender

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

                        # 解析元数据中的性别和年龄
                        age, gender = self.parse_metadata(record.comments)

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
                        self.ages.append(age)
                        self.genders.append(gender)
                        self.record_names.append(record_name)

                        count += 1
                        if count >= self.max_records:
                            # 返回为 NumPy 数组
                            return (np.array(self.ecg_data), np.array(self.labels),
                                    np.array(self.labels_code), np.array(self.ages),
                                    np.array(self.genders), self.record_names)
                    except Exception as e:
                        print(f"Error loading {file}: {e}")
                        continue

        # Ensure the return is NumPy arrays
        return (np.array(self.ecg_data), np.array(self.labels),
                np.array(self.labels_code), np.array(self.ages),
                np.array(self.genders), self.record_names)

    def save_to_csv(self, train_file='../data/train_data.csv', test_file='../data/test_data.csv', test_size=0.2):
        """将预处理数据划分为训练集和测试集并保存为 CSV 文件"""
        # 创建完整的 DataFrame
        df = pd.DataFrame(self.ecg_data, columns=self.feature_columns)
        df['label'] = self.labels
        df['label_code'] = self.labels_code
        df['age'] = self.ages
        df['gender'] = self.genders

        # 划分训练集和测试集
        train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)

        # 保存为 CSV 文件
        train_df.to_csv(train_file, index=False)
        test_df.to_csv(test_file, index=False)
        print(f"Training data saved to {train_file}")
        print(f"Testing data saved to {test_file}")


def main():
    # 数据目录和标签文件路径
    data_dir = "../data/WFDBRecords"  # 数据文件路径
    label_file = "../data/ConditionNames_SNOMED-CT.csv"  # 标签文件路径

    # 创建 ECGDataLoader 实例
    ecg_loader = ECGDataLoader(data_dir, label_file)

    # 加载并处理数据
    ecg_data, labels, labels_code, ages, genders, record_names = ecg_loader.load_and_preprocess_data()

    # 打印数据的基本信息
    print(f"Loaded {len(ecg_data)} ECG records.")
    print(f"Sample features: {ecg_data[0]}")
    print(f"Sample label: {labels[0]}")
    print(f"Sample label code: {labels_code[0]}")
    print(f"Sample age: {ages[0]}")
    print(f"Sample gender: {genders[0]}")

    # 划分并保存数据
    ecg_loader.save_to_csv('../data/train_data.csv', '../data/test_data.csv')


if __name__ == "__main__":
    main()