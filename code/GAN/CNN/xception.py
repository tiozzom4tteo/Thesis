from keras.models import Model
from keras.layers import Input, Conv2D, SeparableConv2D, MaxPooling2D, BatchNormalization, Activation, Dense, GlobalAveragePooling2D, Add
from keras_tuner import HyperModel, RandomSearch
from keras.utils import normalize, to_categorical
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd

# Xception Architecture
def build_xception_model(input_shape, num_classes):
    inputs = Input(shape=input_shape)

    # Entry flow
    def entry_flow(inputs):
        x = Conv2D(32, 3, strides=2, padding='same')(inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv2D(64, 3, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        previous_block_activation = x

        for size in [128, 256, 728]:
            x = Activation('relu')(x)
            x = SeparableConv2D(size, 3, padding='same')(x)
            x = BatchNormalization()(x)

            x = Activation('relu')(x)
            x = SeparableConv2D(size, 3, padding='same')(x)
            x = BatchNormalization()(x)

            x = MaxPooling2D(3, strides=2, padding='same')(x)

            residual = Conv2D(size, 1, strides=2, padding='same')(previous_block_activation)
            x = Add()([x, residual])
            previous_block_activation = x

        return x

    # Middle flow
    def middle_flow(x, num_blocks=8):
        previous_block_activation = x
        for _ in range(num_blocks):
            x = Activation('relu')(x)
            x = SeparableConv2D(728, 3, padding='same')(x)
            x = BatchNormalization()(x)

            x = Activation('relu')(x)
            x = SeparableConv2D(728, 3, padding='same')(x)
            x = BatchNormalization()(x)

            x = Activation('relu')(x)
            x = SeparableConv2D(728, 3, padding='same')(x)
            x = BatchNormalization()(x)

            x = Add()([x, previous_block_activation])
            previous_block_activation = x

        return x

    # Exit flow
    def exit_flow(x):
        previous_block_activation = x

        x = Activation('relu')(x)
        x = SeparableConv2D(728, 3, padding='same')(x)
        x = BatchNormalization()(x)

        x = Activation('relu')(x)
        x = SeparableConv2D(1024, 3, padding='same')(x)
        x = BatchNormalization()(x)

        x = MaxPooling2D(3, strides=2, padding='same')(x)

        residual = Conv2D(1024, 1, strides=2, padding='same')(previous_block_activation)
        x = Add()([x, residual])

        x = Activation('relu')(x)
        x = SeparableConv2D(1536, 3, padding='same')(x)
        x = BatchNormalization()(x)

        x = Activation('relu')(x)
        x = SeparableConv2D(2048, 3, padding='same')(x)
        x = BatchNormalization()(x)

        x = GlobalAveragePooling2D()(x)
        x = Dense(num_classes, activation='softmax')(x)
        return x

    # Build the model
    x = entry_flow(inputs)
    x = middle_flow(x)
    outputs = exit_flow(x)

    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# HyperModel for Xception
class XceptionHyperModel(HyperModel):
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes

    def build(self, hp):
        model = build_xception_model(self.input_shape, self.num_classes)
        return model

# Prepare data
SIZE = 16
categories = ['Adware', 'Backdoor', 'Downloader', 'Ransomware', 'Spyware', 'Trojan', 'Virus']
label_dict = {category: i for i, category in enumerate(categories)}
num_classes = len(categories)

# Directory setup
image_directory = '../../Images/16x16_grayscale_images/'
dataset = []
labels = []

for category in categories:
    path = os.path.join(image_directory, category)
    images = [img for img in os.listdir(path) if img.endswith('.png')]
    for image_name in images:
        img_path = os.path.join(path, image_name)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        dataset.append(np.array(image))
        labels.append(label_dict[category])

# Prepare dataset for model input
dataset = np.array(dataset).reshape(-1, SIZE, SIZE, 1)
labels = np.array(labels)
labels = to_categorical(labels, num_classes=len(categories))

# Split dataset
X_train, X_temp, y_train, y_temp = train_test_split(dataset, labels, test_size=0.20, random_state=0)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=0)

# Normalize data
X_train = normalize(X_train, axis=1)
X_val = normalize(X_val, axis=1)
X_test = normalize(X_test, axis=1)

# Hyperparameter tuning
hypermodel = XceptionHyperModel((SIZE, SIZE, 1), num_classes)

tuner = RandomSearch(
    hypermodel,
    objective='val_accuracy',
    max_trials=2,
    executions_per_trial=1,
    directory='my_dir',
    project_name='malware_classification_tuning'
)

tuner.search(X_train, y_train, epochs=10, validation_data=(X_val, y_val))
best_model = tuner.get_best_models(num_models=1)[0]

# Save the best model
value = 5
os.makedirs(f'{value}/models', exist_ok=True)
best_model.save(f'{value}/models/malware_classification_model_best.h5')

# Train the best model
history = best_model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_val, y_val), shuffle=True)

# Save training results
plt.figure()
plt.plot(history.history['accuracy'], 'y', label='Training Accuracy')
plt.plot(history.history['val_accuracy'], 'r', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig(f'{value}/png/accuracy_xception.png')

# Evaluate and save metrics
y_pred = best_model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)  # Predictions
y_true_classes = np.argmax(y_test, axis=1)  # True classes

# Calculate metrics
report = classification_report(y_true_classes, y_pred_classes, target_names=categories)
conf_matrix = confusion_matrix(y_true_classes, y_pred_classes)
precision = precision_score(y_true_classes, y_pred_classes, average='weighted')
recall = recall_score(y_true_classes, y_pred_classes, average='weighted')
f1 = f1_score(y_true_classes, y_pred_classes, average='weighted')

# Save metrics to a file
output_directory = f'{value}/results'
os.makedirs(output_directory, exist_ok=True)
metrics_file = os.path.join(output_directory, 'evaluation_metrics.txt')

with open(metrics_file, 'w') as file:
    file.write("Classification Report:\n")
    file.write(report + "\n")
    file.write("Confusion Matrix:\n")
    file.write(np.array2string(conf_matrix) + "\n")
    file.write(f"Precision: {precision:.4f}\n")
    file.write(f"Recall: {recall:.4f}\n")
    file.write(f"F1 Score: {f1:.4f}\n")

# Save confusion matrix plot
plt.figure(figsize=(10, 7))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.colorbar()
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.xticks(np.arange(len(categories)), categories, rotation=45)
plt.yticks(np.arange(len(categories)), categories)
plt.tight_layout()
plt.savefig(os.path.join(output_directory, 'confusion_matrix.png'))
plt.close()

print(f"Metrics and confusion matrix saved to {output_directory}")
