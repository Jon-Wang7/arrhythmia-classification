import os
import wfdb
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, butter, filtfilt
from scipy.fft import rfft, rfftfreq
from sklearn.model_selection import train_test_split
import pywt  # 小波变换
import nolds  # 非线性特征，如熵
import signal  # 超时机制

# 超时机制
class TimeoutException(Exception): pass
def timeout_handler(signum, frame):
    raise TimeoutException()

signal.signal(signal.SIGALRM, timeout_handler)

# 带通滤波器
def butter_bandpass_filter(signal, lowcut=0.5, highcut=50.0, fs=500, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

# 提取频域特征
def extract_frequency_features(signal, fs=500):
    fft_values = np.abs(rfft(signal))
    fft_frequencies = rfftfreq(len(signal), 1 / fs)
    low_freq_power = np.sum(fft_values[(fft_frequencies >= 0.04) & (fft_frequencies < 0.15)])
    high_freq_power = np.sum(fft_values[(fft_frequencies >= 0.15) & (fft_frequencies < 0.4)])
    return low_freq_power, high_freq_power

# 提取小波能量特征
def extract_wavelet_features(signal):
    coeffs = pywt.wavedec(signal, 'db4', level=4)
    wavelet_energy = [np.sum(np.square(c)) for c in coeffs[:4]]
    return wavelet_energy

# 提取HRV特征
def extract_hrv_features(r_peaks, fs=500):
    if len(r_peaks) < 2:
        return [0, 0, 0]
    rr_intervals = np.diff(r_peaks) / fs
    mean_rr = np.mean(rr_intervals)
    std_rr = np.std(rr_intervals)
    rmssd = np.sqrt(np.mean(np.square(np.diff(rr_intervals))))
    return [mean_rr, std_rr, rmssd]

# 提取熵特征
def extract_entropy(signal):
    try:
        return nolds.sampen(signal)
    except:
        return 0

# 组合所有特征
def extract_features(segment, low_freq_power, high_freq_power, r_peaks):
    mean = np.mean(segment)
    std_dev = np.std(segment)
    max_val = np.max(segment)
    min_val = np.min(segment)
    entropy = extract_entropy(segment)
    wavelet_features = extract_wavelet_features(segment)
    hrv_features = extract_hrv_features(r_peaks)
    return [mean, std_dev, max_val, min_val, low_freq_power, high_freq_power, entropy] + wavelet_features + hrv_features

# 获取标签
def get_label_from_dx(dx_field):
    if dx_field:
        dx_codes = dx_field.split(":")[1].strip()
        label_code = int(dx_codes.split(",")[0].strip())
        return label_code
    else:
        return 'Unknown'

# 数据预处理
def preprocess_ecg_data(data_dir, output_dir, sample_length=260, max_samples=20000):
    signals, labels = [], []
    sample_count = 0

    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".mat"):
                record_path = os.path.join(root, file[:-4])
                try:
                    print(f"Processing file: {file}")
                    record = wfdb.rdrecord(record_path)
                    ecg_signal = record.p_signal[:, 0]

                    # 处理NaN值
                    ecg_signal = np.nan_to_num(ecg_signal, nan=0.0, posinf=0.0, neginf=0.0)

                    # 信号有效性检查
                    if len(ecg_signal) < sample_length or np.all(ecg_signal == 0):
                        print(f"Invalid signal in file: {file}, skipping...")
                        continue

                    # 频域特征
                    low_freq_power, high_freq_power = extract_frequency_features(ecg_signal)

                    # 滤波
                    ecg_signal = butter_bandpass_filter(ecg_signal)

                    # R峰检测
                    signal.alarm(10)  # 设置超时
                    r_peaks, _ = find_peaks(ecg_signal, distance=150, prominence=0.5)
                    signal.alarm(0)  # 清除超时

                    if len(r_peaks) == 0:
                        print(f"No R-peaks detected in file: {file}, skipping...")
                        continue

                    # 提取每个R峰附近的段
                    for peak in r_peaks:
                        start, end = peak - sample_length // 2, peak + sample_length // 2
                        if start >= 0 and end < len(ecg_signal):
                            segment = ecg_signal[start:end]

                            if segment.size == 0 or np.all(segment == 0):
                                print(f"Empty segment in file: {file}, skipping...")
                                continue

                            features = extract_features(segment, low_freq_power, high_freq_power, r_peaks)
                            signals.append(features)

                            # 获取标签
                            dx_field = record.comments[2] if len(record.comments) > 2 else ""
                            label = get_label_from_dx(dx_field)
                            labels.append(label)

                            sample_count += 1
                            if sample_count >= max_samples:
                                save_train_test_split(signals, labels, output_dir)
                                return

                except TimeoutException:
                    print(f"Timeout processing file: {file}")
                except Exception as e:
                    print(f"Error processing file {file}: {e}")

    save_train_test_split(signals, labels, output_dir)

# 划分训练和测试集
def save_train_test_split(signals, labels, output_dir, test_size=0.2):
    columns = ['mean', 'std_dev', 'max', 'min', 'low_freq_power', 'high_freq_power', 'entropy',
               'wavelet_1', 'wavelet_2', 'wavelet_3', 'wavelet_4', 'mean_rr', 'std_rr', 'rmssd']
    df = pd.DataFrame(signals, columns=columns)
    df['label'] = labels

    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)
    print("Data saved successfully!")

if __name__ == "__main__":
    data_dir = "../data/WFDBRecords"
    output_dir = "../data/"
    preprocess_ecg_data(data_dir, output_dir)