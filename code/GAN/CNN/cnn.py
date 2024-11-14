import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from keras.utils import normalize, to_categorical
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_score, recall_score
import pandas as pd

# Directory setup
image_directory = '../../Images/16x16_grayscale_images/'
SIZE = 150
dataset = []
labels = []

# Class label assignment
categories = ['Adware', 'Backdoor', 'Downloader', 'Ransomware', 'Spyware', 'Trojan', 'Virus']
label_dict = {category: i for i, category in enumerate(categories)}

# Load and process images
for category in categories:
    path = os.path.join(image_directory, category)
    images = [img for img in os.listdir(path) if img.endswith('.png')]
    for image_name in images:
        img_path = os.path.join(path, image_name)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (SIZE, SIZE))
        dataset.append(np.array(image))
        labels.append(label_dict[category])

dataset = np.array(dataset).reshape(-1, SIZE, SIZE, 1)
labels = np.array(labels)
labels = to_categorical(labels, num_classes=len(categories))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(dataset, labels, test_size=0.20, random_state=0)
X_train = normalize(X_train, axis=1)
X_test = normalize(X_test, axis=1)

# Model definition
model = Sequential([
    Conv2D(32, (3, 3), input_shape=(SIZE, SIZE, 1), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(len(categories), activation='softmax')
])

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())

# Model 
######################
history = model.fit(X_train, y_train, batch_size=64, epochs=100, validation_data=(X_test, y_test), shuffle=False)

# Save the model
model.save('models/malware_classification_model.h5')

# Load model
model = load_model('models/malware_classification_model.h5')

# Evaluate model
_, acc = model.evaluate(X_test, y_test)
y_pred = (model.predict(X_test) >= 0.5).astype(int) ###################
precision = precision_score(y_test.argmax(axis=1), y_pred.argmax(axis=1), average='macro')
recall = recall_score(y_test.argmax(axis=1), y_pred.argmax(axis=1), average='macro')

# Save evaluation metrics to text file
with open('evaluation_metrics.txt', 'w') as f:
    f.write(f"Accuracy = {acc * 100:.2f}%\n")
    f.write(f"Precision = {precision:.2f}\n")
    f.write(f"Recall = {recall:.2f}\n")

print(f"Accuracy = {acc * 100:.2f}%")

# Confusion matrix
y_pred = (model.predict(X_test) >= 0.5).astype(int)
cm = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
np.savetxt('confusion_matrix.txt', cm, fmt='%d')
print(cm)

# ROC curve
y_preds = model.predict(X_test).ravel()
fpr, tpr, thresholds = roc_curve(y_test.ravel(), y_preds)
roc_auc = auc(fpr, tpr)
print("Area under curve, AUC =", roc_auc)

plt.figure()
plt.plot([0, 1], [0, 1], 'y--')
plt.plot(fpr, tpr, marker='.')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC Curve')
plt.savefig('roc_curve.png')

# Training and validation loss
plt.plot(history.history['loss'], 'y', label='Training loss')
plt.plot(history.history['val_loss'], 'r', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('loss.png')

# Training and validation accuracy
plt.plot(history.history['accuracy'], 'y', label='Training Accuracy')
plt.plot(history.history['val_accuracy'], 'r', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('accuracy.png')

# Ideal ROC threshold
i = np.arange(len(tpr))
roc = pd.DataFrame({'tf': pd.Series(tpr - (1 - fpr), index=i), 'thresholds': pd.Series(thresholds, index=i)})
ideal_thresh = roc.iloc[(roc.tf - 0).abs().argsort()[:1]]
print("Ideal threshold is:", ideal_thresh['thresholds'].values[0])




#Finire con cnn e partire con la gan

