import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from keras.utils import normalize, to_categorical
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
import pandas as pd
from keras_tuner import HyperModel, RandomSearch
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score


# Tune the model with keras tuner and try to tune the model definition

# Directory setup
image_directory = '../../Images/16x16_grayscale_images/'
SIZE = 16  
dataset = []
labels = []

value = 5
os.makedirs(f'{value}/txt', exist_ok=True)
os.makedirs(f'{value}/png', exist_ok=True)


# Class label assignment
categories = ['Adware', 'Backdoor', 'Downloader', 'Ransomware', 'Spyware', 'Trojan', 'Virus']
label_dict = {category: i for i, category in enumerate(categories)}
num_classes = len(categories)
# Load and process images (no resizing or data augmentation)
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

# Split dataset: 80% training, 10% validation, 10% testing
X_train, X_temp, y_train, y_temp = train_test_split(dataset, labels, test_size=0.20, random_state=0)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=0)

# Normalize data
X_train = normalize(X_train, axis=1)
X_val = normalize(X_val, axis=1)
X_test = normalize(X_test, axis=1)

# Model definition 3
class MalwareModelHyperModel(HyperModel):
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def build(self, hp):
        model = Sequential([
            Conv2D(hp.Int('conv_1_filters', 32, 128, step=32), (3, 3), input_shape=(SIZE, SIZE, 1), padding='same', activation='relu'),
            MaxPooling2D(2, 2),
            Conv2D(hp.Int('conv_2_filters', 32, 64, step=32), (3, 3), padding='same', activation='relu'),
            MaxPooling2D(2, 2),
            Conv2D(hp.Int('conv_3_filters', 32, 64, step=32), (3, 3), padding='same', activation='relu'),
            MaxPooling2D(2, 2),
            Flatten(),
            Dense(hp.Int('dense_units', 32, 128, step=32), activation='relu'),
            Dropout(hp.Float('dropout', 0.0, 0.5, step=0.1)),
            Dense(self.num_classes, activation='softmax')
        ])
        # model = Sequential([
        #     Conv2D(hp.Int('conv_1_filters', 32, 128, step=32), (3, 3), input_shape=(SIZE, SIZE, 1), padding='same', activation='relu'),
        #     MaxPooling2D(2, 2),
        #     Conv2D(hp.Int('conv_2_filters', 32, 64, step=32), (3, 3), padding='same', activation='relu'),
        #     MaxPooling2D(2, 2),
        #     Conv2D(hp.Int('conv_3_filters', 32, 64, step=32), (3, 3), padding='same', activation='relu'),
        #     MaxPooling2D(2, 2),
        #     Flatten(),
        #     Dense(hp.Int('dense_units', 64, 128, step=32), activation='relu'),
        #     Dropout(hp.Float('dropout', 0.0, 0.2, step=0.1)),
        #     Dense(hp.Int('dense_units', 32, 64, step=32), activation='relu'),
        #     Dropout(hp.Float('dropout', 0.0, 0.5, step=0.1)),
        #     Dense(hp.Int('dense_units', 32, 32, step=32), activation='relu'),
        #     Dropout(hp.Float('dropout', 0.0, 0.2, step=0.1)),
        #     Dense(self.num_classes, activation='softmax')
        # ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

hypermodel = MalwareModelHyperModel(num_classes)

tuner = RandomSearch(
    hypermodel,
    objective='val_accuracy',
    max_trials=10,
    executions_per_trial=5,
    directory='my_dir',
    project_name='malware_classification_tuning'
)

tuner.search(X_train, y_train, epochs=30, validation_data=(X_val, y_val))
best_model = tuner.get_best_models(num_models=1)[0]
best_model.save(f'{value}/models/malware_classification_model_best.h5')

model = best_model

print(model.summary())

# Model training
history = model.fit(X_train, y_train, batch_size=32, epochs=100, validation_data=(X_val, y_val), shuffle=True)

# Save the model
model.save(f'{value}/models/malware_classification_model.h5')

# Load model
model = load_model(f'{value}/models/malware_classification_model.h5')

# Valutazione del modello
_, acc = model.evaluate(X_test, y_test)
y_pred = model.predict(X_test).argmax(axis=1)
y_true = y_test.argmax(axis=1)

# Calcolo delle metriche per micro e macro average
micro_precision = precision_score(y_true, y_pred, average='micro')
macro_precision = precision_score(y_true, y_pred, average='macro')
micro_recall = recall_score(y_true, y_pred, average='micro')
macro_recall = recall_score(y_true, y_pred, average='macro')
micro_f1 = f1_score(y_true, y_pred, average='micro')
macro_f1 = f1_score(y_true, y_pred, average='macro')

# Generazione del report di classificazione
report = classification_report(y_true, y_pred, target_names=categories, output_dict=True)

# Aggiunta delle metriche micro avg manualmente nel dizionario di output
report['micro avg'] = {
    'precision': micro_precision,
    'recall': micro_recall,
    'f1-score': micro_f1,
    'support': len(y_true)
}

# Conversione del report in formato testo per salvarlo nel file
report_text = f"Accuracy = {acc * 100:.2f}%\n\n"
report_text += "Classification Report:\n"
report_text += classification_report(y_true, y_pred, target_names=categories)
report_text += f"\nMicro Precision: {micro_precision:.2f}\n"
report_text += f"Macro Precision: {macro_precision:.2f}\n"
report_text += f"Micro Recall: {micro_recall:.2f}\n"
report_text += f"Macro Recall: {macro_recall:.2f}\n"
report_text += f"Micro F1 Score: {micro_f1:.2f}\n"
report_text += f"Macro F1 Score: {macro_f1:.2f}\n"

# Salvataggio del report con micro e macro avg
with open(f'{value}/txt/evaluation_metrics.txt', 'w') as f:
    f.write(report_text)


# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
np.savetxt(f'{value}/txt/confusion_matrix.txt', cm, fmt='%d')

# Training and validation accuracy plot
plt.figure()
plt.plot(history.history['accuracy'], 'y', label='Training Accuracy')
plt.plot(history.history['val_accuracy'], 'r', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig(f'{value}/png/accuracy.png')

# Training and validation loss
plt.figure()
plt.plot(history.history['loss'], 'y', label='Training loss')
plt.plot(history.history['val_loss'], 'r', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig(f'{value}/png/loss.png')

# # ROC curve and AUC (for multi-class)
# y_preds = model.predict(X_test).ravel()
# fpr, tpr, thresholds = roc_curve(y_test.ravel(), y_preds)
# roc_auc = auc(fpr, tpr)
# print("Area under curve, AUC =", roc_auc)

# plt.figure()
# plt.plot([0, 1], [0, 1], 'y--')
# plt.plot(fpr, tpr, marker='.')
# plt.xlabel('False positive rate')
# plt.ylabel('True positive rate')
# plt.title('ROC Curve')
# plt.savefig('roc_curve.png')



# Model definition 1
# model = Sequential([
#     Conv2D(128, (3, 3), input_shape=(SIZE, SIZE, 1),
#            padding='same', activation='relu'),
#     MaxPooling2D(2, 2),
#     Conv2D(64, (3, 3), #input_shape=(SIZE//2, SIZE//2, 1),
#            padding='same', activation='relu'),
#     MaxPooling2D(2, 2),

#     Conv2D(32, (3, 3), #input_shape=(SIZE//4, SIZE//4, 1),
#            padding='same',  activation='relu'),
#     MaxPooling2D(2, 2),
#     Flatten(),
#     Dense(64, activation='relu'),
#     Dropout(0.5),
#     Dense(len(categories), activation='softmax')
# ])

# Model definition 2
# model = Sequential([
#     Conv2D(128, (3, 3), input_shape=(SIZE, SIZE, 1),
#            padding='same', activation='relu'),
#     MaxPooling2D(2, 2),
#     Conv2D(64, (3, 3), input_shape=(SIZE//2, SIZE//2, 1),
#            padding='same', activation='relu'),
#     MaxPooling2D(2, 2),

#     Conv2D(32, (3, 3), input_shape=(SIZE//4, SIZE//4, 1),
#            padding='same',  activation='relu'),
#     MaxPooling2D(2, 2),
#     Flatten(),
#     Dense(64, activation='relu'),
#     Dropout(0.9),
#     Dense(len(categories), activation='softmax')
# ])