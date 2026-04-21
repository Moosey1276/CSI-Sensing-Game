from hampel import hampel
import numpy as np
from scipy.signal import savgol_filter
import pandas as pd
import os
import matplotlib.pyplot as plt
import re
from math import sqrt, atan2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout, Attention, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
import seaborn as sns
import optuna

# Specify the path to your main folder
main_folder_path = ('../')

activity_dataframes = []
label_list = []

for file_name in os.listdir(main_folder_path):
    if file_name.endswith('.csv'):
        print(file_name)
        file_path = os.path.join(main_folder_path, file_name)
        label_list.append(file_name.replace('.csv', ''))
        df = pd.read_csv(file_path, on_bad_lines='skip')
        activity_dataframes.append(df)

print(len(activity_dataframes))
for dataframe in activity_dataframes:
    print(dataframe.shape)

min_length = min(len(df) for df in activity_dataframes)
print(f"\nShortest DataFrame length: {min_length}")

trimmed_dataframes = [df.iloc[:min_length].copy() for df in activity_dataframes]

amp_dataframes = []

for df in activity_dataframes:
    data = []

    for _, row in df.iterrows():
        imaginary = []
        real = []
        amplitudes = []

        if isinstance(row['CSI_DATA'], str):
            try:
                csi_string = re.findall(r"\[(.*)\]", row['CSI_DATA'])[0]

                csi_raw = []
                for x in csi_string.split(" "):
                    try:
                        csi_raw.append(int(x))
                    except ValueError:
                        csi_raw.append(0)

                for i in range(len(csi_raw)):
                    if i % 2 == 0:
                        imaginary.append(csi_raw[i])
                    else:
                        real.append(csi_raw[i])

                for i in range(int(len(csi_raw) / 2)):
                    amp = sqrt(imaginary[i] ** 2 + real[i] ** 2)
                    if amp < 40.0:
                        amplitudes.append(amp)
                    else:
                        if amp != 105.60303025955268:
                            print("WARNING! TOO HIGH AMPLITUDE")
                            print(amp)

                        amplitudes.append(35.0)

                if not any(np.isnan(amplitudes)):
                    data.append(amplitudes)
                else:
                    print("Skipping row with NaN amplitude.")
            except (IndexError, ValueError) as e:
                print(f"Skipping malformed row: {e}")
                continue
        else:
            print(f"Skipping row due to invalid CSI_DATA type: {row['CSI_DATA']}")
            continue

    clean_data = [row for row in data if len(row) == 64 and not any(np.isnan(row))]
    temp_df = pd.DataFrame(clean_data)
    amp_dataframes.append(temp_df)

print(label_list)

plt.figure(figsize=(8, 6))
plt.plot(amp_dataframes[1][50],color='g')
plt.xlabel('Index')
plt.ylabel('Amplitude')
plt.title('Raw CSI Amplitude Data')
plt.show()

for i in range (0, 64):
  print(amp_dataframes[2][i])

denoised_dataframes = []
for amplitude in amp_dataframes:
    filtered_data = pd.DataFrame()
    for col in amplitude.columns:
      col_series = amplitude[col]
      # Hampel filter
      hampel_filtered = hampel(col_series, window_size=10)
      # Savitzky-Golay filter
      sg_filtered = savgol_filter(hampel_filtered.filtered_data, window_length=10, polyorder=3) #GPT reccomends to change window lenght to 10
      filtered_data[col] = sg_filtered
    denoised_dataframes.append(filtered_data)

for i, df in enumerate(denoised_dataframes):
    print(f"[Filter Check] File {i}: shape = {df.shape}, NaNs = {df.isna().sum().sum()}, Any NaN rows: {df.isna().any(axis=1).sum()}")

plt.figure(figsize=(8, 6))
plt.plot(denoised_dataframes[1][50],color='g')
plt.xlabel('Index')
plt.ylabel('Amplitude')
plt.title('Denoise CSI Amplitude Data')
plt.show()

columns_to_drop = [0,1,2,3,4,5,32,59,60,61,62,63]
for df in denoised_dataframes:
    df.drop(df.columns[columns_to_drop], axis=1,inplace=True)

segment_dataframes = []
labels = []
for i, df in enumerate(denoised_dataframes):
  df_len = len(df)
  segment_len = (df_len//50)*50
  rows_to_skip = len(df) - segment_len
  rounded_df = df.iloc[rows_to_skip:]
  segment_df = np.array_split(rounded_df, range(50, len(rounded_df), 50))
  for segment in segment_df:
    segment_dataframes.append(segment)
    labels.append(label_list[i])

print(len(segment_dataframes))
print(len(labels))

def visualize_pose_comparison(segment_dataframes, labels, num_samples=3):
    poses = ['crouching', 'ski_jump', 'standing', 'x_pose']
    fig, axes = plt.subplots(4, num_samples, figsize=(5 * num_samples, 8))

    for row, pose in enumerate(poses):
        count = 0
        for i, (segment, label) in enumerate(zip(segment_dataframes, labels)):
            if label == pose:
                ax = axes[row][count]
                sns.heatmap(segment.T, cmap='viridis', ax=ax)
                ax.set_title(f"{pose} - Sample {count + 1}")
                ax.set_xlabel("Packet Index")
                ax.set_ylabel("Subcarrier Index")
                ax.invert_yaxis()  # Make low subcarrier index at bottom, high at top
                count += 1
                if count >= num_samples:
                    break

    plt.tight_layout()
    plt.show()

visualize_pose_comparison(segment_dataframes, labels, num_samples=4)

labels = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(segment_dataframes, labels, test_size=0.3, random_state=42)

X_train = np.array(X_train)
X_train = X_train.astype('float32')
X_train /= 255
X_test = np.array(X_test)
X_test = X_test.astype('float32')
X_test /= 255

label_encoder = LabelEncoder()
label_encoder.fit(y_train)
y_train_encoded = label_encoder.transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

print(np.isnan(X_train).sum(), np.isinf(X_train).sum())
print(np.unique(y_train_encoded))
X_train = X_train[..., np.newaxis]
X_test = X_test[..., np.newaxis]
print(X_train.shape)

def objective(trial):
    n_conv_blocks = trial.suggest_int("n_conv_blocks", 1, 4)

    raw_filters = []
    raw_kernels = []
    for i in range(n_conv_blocks):
        raw_filters.append(trial.suggest_categorical(f"filters_{i}", [32, 64, 96, 128]))
        raw_kernels.append(trial.suggest_categorical(f"kernel_{i}", [3, 5, 7]))

    filters = [max(raw_filters[:i+1]) for i in range(len(raw_filters))]
    kernels = [min(raw_kernels[:i+1]) for i in range(len(raw_kernels))]
    print(filters)
    print(kernels)

    n_dense_layers = trial.suggest_int("n_dense_layers", 1, 4)
    dense_units = []
    for i in range(n_dense_layers):
        dense_units.append(trial.suggest_categorical(f"dense_units_{i}", [64, 128, 256]))

    dropout_rate = trial.suggest_float("dropout", 0.2, 0.5)

    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    batch = trial.suggest_categorical("batch_size", [8, 16, 32])

    epochs = 30

    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(50, 52, 1)))

    for i in range(n_conv_blocks):
        model.add(Conv2D(filters[i], kernels[i], activation="relu", padding="same"))
        model.add(MaxPooling2D(pool_size=2))

    model.add(Flatten())

    for i in range(n_dense_layers):
        model.add(Dense(dense_units[i], activation="relu"))
        model.add(Dropout(dropout_rate))

    model.add(Dense(4, activation="softmax"))

    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    history = model.fit(
        X_train, y_train_encoded,
        validation_data=(X_test, y_test_encoded),
        epochs=epochs,
        batch_size=batch,
        verbose=0  # mute training logs for speed
    )

    val_acc = max(history.history["val_accuracy"])
    return val_acc

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)

print("Best trial:")
trial = study.best_trial
print("  Validation Accuracy:", trial.value)
print("  Best Params:", trial.params)