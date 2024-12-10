import os
import wfdb
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, butter, filtfilt
from scipy.fft import rfft, rfftfreq
from sklearn.model_selection import train_test_split

# 提取频域特征（低频和高频功率）
def extract_frequency_features(signal, fs=500):
    fft_values = np.abs(rfft(signal))  # 傅里叶变换
    fft_frequencies = rfftfreq(len(signal), 1 / fs)

    # 提取低频和高频功率
    low_freq_power = np.sum(fft_values[(fft_frequencies >= 0.04) & (fft_frequencies < 0.15)])
    high_freq_power = np.sum(fft_values[(fft_frequencies >= 0.15) & (fft_frequencies < 0.4)])

    return low_freq_power, high_freq_power

# 带通滤波器
def butter_bandpass_filter(signal, lowcut=0.5, highcut=50.0, fs=500, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

# 特征提取函数（包括时域特征）
def extract_features(signal, low_freq_power, high_freq_power):
    mean = np.mean(signal)
    std_dev = np.std(signal)
    max_val = np.max(signal)
    min_val = np.min(signal)

    # 返回包括频域特征的时域特征
    return [mean, std_dev, max_val, min_val, low_freq_power, high_freq_power]

# 解析 DX 字段，取第一个标签
def get_label_from_dx(dx_field):
    if dx_field:
        dx_codes = dx_field.split(":")[1].strip()
        label_code = int(dx_codes.split(",")[0].strip())
        return label_code
    else:
        return 'Unknown'

# 预处理并保存数据
def preprocess_ecg_data(data_dir, output_dir, sample_length=260, max_samples=5000):
    signals, labels = [], []
    sample_count = 0

    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".mat"):
                record_path = os.path.join(root, file[:-4])
                try:
                    record = wfdb.rdrecord(record_path)
                    signal = record.p_signal[:, 0]  # 使用 Lead II

                    # 提取频域特征
                    low_freq_power, high_freq_power = extract_frequency_features(signal)
                    print("low_freq_power:", low_freq_power, "high_freq_power:", high_freq_power)

                    # 带通滤波器
                    signal = butter_bandpass_filter(signal)

                    # 检测 R 峰
                    r_peaks, _ = find_peaks(signal, distance=150, prominence=0.5)
                    if len(r_peaks) == 0:
                        continue

                    for peak in r_peaks:
                        if peak - sample_length // 2 >= 0 and peak + sample_length // 2 < len(signal):
                            segment = signal[peak - sample_length // 2: peak + sample_length // 2]
                            features = extract_features(segment, low_freq_power, high_freq_power)

                            signals.append(features)

                            # 获取第一个 DX 标签
                            dx_field = record.comments[2]  # 假设 DX 信息位于评论的第 3 项
                            label = get_label_from_dx(dx_field)
                            labels.append(label)

                            sample_count += 1
                            if sample_count >= max_samples:
                                save_train_test_split(signals, labels, output_dir)
                                return

                except Exception as e:
                    print(f"Error processing file {file}: {e}")
                    continue

    if len(signals) == 0:
        raise ValueError("No valid signals processed. Check your data and parameters.")

    save_train_test_split(signals, labels, output_dir)

# 划分训练集和测试集并保存
def save_train_test_split(signals, labels, output_dir, test_size=0.2):
    df = pd.DataFrame(signals, columns=['mean', 'std_dev', 'max', 'min', 'low_freq_power', 'high_freq_power'])
    df['label'] = labels

    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)

    train_file = os.path.join(output_dir, "train.csv")
    test_file = os.path.join(output_dir, "test.csv")

    train_df.to_csv(train_file, index=False)
    test_df.to_csv(test_file, index=False)

    print(f"Training data saved to {train_file}")
    print(f"Testing data saved to {test_file}")

if __name__ == "__main__":
    data_dir = "../data/WFDBRecords"
    output_dir = "../data/"
    preprocess_ecg_data(data_dir, output_dir)