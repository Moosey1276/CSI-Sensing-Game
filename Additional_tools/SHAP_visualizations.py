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
from tensorflow.keras.layers import Dense, Dropout, Attention
from tensorflow.keras.losses import SparseCategoricalCrossentropy
import seaborn as sns
import shap


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

print("\nAFTER TRIMMING:")
for df in trimmed_dataframes:
    print(df.shape)

amp_dataframes = []
above_100_count = 0
above_500_count = 0

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
                            if amp > 500:
                                above_500_count += 1
                            if amp > 100:
                                above_100_count += 1

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
print(amp_dataframes)
print(above_100_count)
print(above_500_count)

plt.figure(figsize=(8, 6))
plt.plot(amp_dataframes[1][50], color='g')
plt.xlabel('Index')
plt.ylabel('Amplitude')
plt.title('Raw CSI Amplitude Data')
plt.show()

denoised_dataframes = []
for amplitude in amp_dataframes:
    filtered_data = pd.DataFrame()
    for col in amplitude.columns:
        col_series = amplitude[col]
        # Hampel filter
        hampel_filtered = hampel(col_series, window_size=10)
        # Savitzky-Golay filter
        sg_filtered = savgol_filter(hampel_filtered.filtered_data, window_length=10,
                                    polyorder=3)  # GPT reccomends to change window lenght to 10
        filtered_data[col] = sg_filtered
    denoised_dataframes.append(filtered_data)

for i, df in enumerate(denoised_dataframes):
    print(f"[Filter Check] File {i}: shape = {df.shape}, NaNs = {df.isna().sum().sum()}, Any NaN rows: {df.isna().any(axis=1).sum()}")

plt.figure(figsize=(8, 6))
plt.plot(denoised_dataframes[1][50], color='g')
plt.xlabel('Index')
plt.ylabel('Amplitude')
plt.title('Denoise CSI Amplitude Data')
plt.show()

columns_to_drop = [0,1,2,3,4,5,32,59,60,61,62,63]
for df in denoised_dataframes:
    df.drop(df.columns[columns_to_drop], axis=1, inplace=True)

segment_dataframes = []
labels = []
for i, df in enumerate(denoised_dataframes):
    df_len = len(df)
    segment_len = (df_len // 50) * 50
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
                ax.invert_yaxis()
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
X_train = X_train[..., np.newaxis]  # Adds channel dimension
X_test = X_test[..., np.newaxis]
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# Define the CNN model
model = keras.Sequential([
    keras.layers.Input(shape=(50, 52, 1)),
    keras.layers.Conv2D(32, 7, activation='relu'),
    keras.layers.MaxPooling2D(2),
    keras.layers.Conv2D(96, 5, activation='relu'),
    keras.layers.MaxPooling2D(2),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.25),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.25),
    keras.layers.Dense(4, activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

history = model.fit(
    X_train, y_train_encoded,
    epochs=30,
    batch_size=16,
    validation_data=(X_test, y_test_encoded)
)

print("\nStarting SHAP analysis...")

def f(x):
    tmp = x.copy()
    return model(tmp)

print("Creating SHAP Image masker for shape (50, 52)...")
masker_blur = shap.maskers.Image("blur(50,52)", X_train[0].shape)

print("Creating SHAP Explainer...")
explainer = shap.Explainer(f, masker_blur, output_names=label_encoder.classes_)

class_names = label_encoder.classes_

manual_positions = {
    0: [7, 9, 10],
    1: [0, 1, 2],
    2: [0, 1, 3],
    3: [0, 1, 2]
}

ind_to_explain = []

for class_index, class_name in enumerate(class_names):
    all_indices_for_class = np.where(y_train_encoded == class_index)[0]
    print(f"Class '{class_name}': total samples = {len(all_indices_for_class)}")

    if class_index in manual_positions:
        pos_list = manual_positions[class_index]

        valid = [
            all_indices_for_class[pos]
            for pos in pos_list
            if pos < len(all_indices_for_class)
        ]

        missing = len(pos_list) - len(valid)
        if missing > 0:
            print(f" - WARNING: {missing} positions out of range for '{class_name}'")

        ind_to_explain.extend(valid)
        print(f" - Using indices {valid} for '{class_name}'")
    else:
        print(f" - WARNING: No manual positions provided for '{class_name}'")

ind_to_explain = np.array(ind_to_explain)

samples_to_explain = X_train[ind_to_explain]

true_labels = label_encoder.inverse_transform(y_train_encoded[ind_to_explain])
print(f"\nExplaining {len(ind_to_explain)} total samples...")
print(f"True labels of samples: {true_labels}")

print("Calculating SHAP values (this may take a while)...")
shap_values = explainer(samples_to_explain, max_evals=5000, batch_size=50)
print("shap_values.values shape:", shap_values.values.shape)
print("shap_values min, max:", shap_values.values.min(), shap_values.values.max())
preds = model.predict(samples_to_explain)
print("pred shape:", preds.shape)
print("pred probs (first few):", preds[:len(ind_to_explain)])
print("pred labels:", np.argmax(preds, axis=1))
print("true labels encoded:", y_train_encoded[ind_to_explain])

print("Generating custom SHAP plots with true labels...")

num_samples = len(ind_to_explain)
class_names = label_encoder.classes_
num_classes = len(class_names)

fig, axes = plt.subplots(nrows=num_samples,
                         ncols=num_classes + 1,
                         figsize=(15, num_samples * 3))

if num_samples == 1:
    axes = np.array([axes])

for i in range(num_samples):
    ax = axes[i, 0]
    ax.imshow(samples_to_explain[i].squeeze(), cmap='viridis')
    ax.set_title(f"True: {true_labels[i]}", fontsize=20)
    ax.axis('off')

    shap_sample = shap_values.values[i, ..., 0, :]  # shape: (H, W, num_classes)
    v_max = np.abs(shap_sample).max()

    for j in range(num_classes):
        ax = axes[i, j + 1]

        shap_data = shap_values.values[i, ..., 0, j]

        im = ax.imshow(shap_data,
                  cmap='bwr',
                  vmin=-v_max,
                  vmax=v_max)

        if i == 0:
            ax.set_title(f"SHAP for: {class_names[j]}", fontsize=20)
        ax.axis('off')

plt.tight_layout()
plt.savefig("SHAP_heatmap.pdf")
plt.show()
