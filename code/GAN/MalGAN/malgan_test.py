import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical, normalize
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def build_generator(latent_dim):
    noise = Input(shape=(latent_dim,))
    x = Dense(256, activation="relu")(noise)
    x = Dense(16 * 16, activation="softmax")(x)  # Output flattened image
    img = Reshape((16, 16, 1))(x)
    model = Model(inputs=noise, outputs=img, name="generator")
    return model

def build_substitute_detector(input_shape, num_classes):
    input_img = Input(shape=input_shape)
    x = Flatten()(input_img)
    x = Dense(128, activation="relu")(x)
    x = Dense(64, activation="relu")(x)
    output = Dense(num_classes, activation="softmax")(x)
    model = Model(inputs=input_img, outputs=output, name="substitute_detector")
    # model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_models(generator, substitute_detector, blackbox_model, X_train, y_train, X_val, y_val, latent_dim, num_classes, epochs=50, batch_size=32):
    # Creazione dei file per salvare le metriche
    with open("blackbox_metrics.txt", "w") as bb_file, open("substitute_metrics.txt", "w") as sub_file:
        # Intestazioni per metriche estese
        bb_file.write("Epoch\tLoss\tAccuracy\tPrecision\tRecall\tF1-Score\n")
        sub_file.write("Epoch\tLoss\tAccuracy\tPrecision\tRecall\tF1-Score\n")

    # Loop di addestramento
    for epoch in range(epochs):
        # 1. Addestramento del substitute detector su dati reali
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        real_images = X_train[idx]
        real_labels = y_train[idx]

        # Aggiungere dimensione del canale per immagini reali
        real_images = np.expand_dims(real_images, axis=-1)
        substitute_loss, substitute_acc = substitute_detector.train_on_batch(real_images, real_labels)

        # 2. Generazione di immagini false
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        fake_images = generator.predict(noise)

        # 3. Ottenere previsioni per immagini false dal substitute detector
        fake_predictions = substitute_detector.predict(fake_images)

        # 4. Calcolo della perdita per il generatore
        # Obiettivo: fare in modo che il substitute detector classifichi le immagini false come una distribuzione uniforme
        uniform_labels = np.ones_like(fake_predictions) / num_classes  # Distribuzione uniforme
        generator_loss = tf.reduce_mean(
            tf.keras.losses.categorical_crossentropy(uniform_labels, fake_predictions)
        )

        # 5. Aggiornamento del generatore
        with tf.GradientTape() as tape:
            fake_images = generator(noise)  # Ricrea le immagini false
            fake_predictions = substitute_detector(fake_images)  # Previsioni sulle immagini false
            generator_loss = tf.reduce_mean(
                tf.keras.losses.categorical_crossentropy(uniform_labels, fake_predictions)
            )
        gradients = tape.gradient(generator_loss, generator.trainable_variables)
        generator.optimizer.apply_gradients(zip(gradients, generator.trainable_variables))

        # 6. Validazione e salvataggio delle metriche ogni 10 epoche
        if epoch % 10 == 0:
            # Validazione del substitute detector
            val_images = np.expand_dims(X_val, axis=-1)
            val_loss, val_acc = substitute_detector.evaluate(val_images, y_val, verbose=0)

            # Calcolo delle previsioni e delle metriche dettagliate
            val_predictions = substitute_detector.predict(val_images)
            val_pred_classes = np.argmax(val_predictions, axis=1)
            val_true_classes = np.argmax(y_val, axis=1)

            substitute_precision = precision_score(val_true_classes, val_pred_classes, average="macro")
            substitute_recall = recall_score(val_true_classes, val_pred_classes, average="macro")
            substitute_f1 = f1_score(val_true_classes, val_pred_classes, average="macro")

            # Validazione del blackbox model
            blackbox_loss, blackbox_acc = blackbox_model.evaluate(val_images, y_val, verbose=0)
            blackbox_predictions = blackbox_model.predict(val_images)
            blackbox_pred_classes = np.argmax(blackbox_predictions, axis=1)

            blackbox_precision = precision_score(val_true_classes, blackbox_pred_classes, average="macro")
            blackbox_recall = recall_score(val_true_classes, blackbox_pred_classes, average="macro")
            blackbox_f1 = f1_score(val_true_classes, blackbox_pred_classes, average="macro")

            # Log dei risultati
            print(f"Epoch {epoch}: Substitute Detector -> Loss: {val_loss}, Accuracy: {val_acc}")
            print(f"Epoch {epoch}: Blackbox Model -> Loss: {blackbox_loss}, Accuracy: {blackbox_acc}")
            print(f"Generator Loss: {generator_loss.numpy()}")

            # Salvataggio delle metriche estese nei file
            with open("substitute_metrics.txt", "a") as sub_file:
                sub_file.write(f"{epoch}\t{val_loss:.6f}\t{val_acc:.6f}\t{substitute_precision:.6f}\t{substitute_recall:.6f}\t{substitute_f1:.6f}\n")

            with open("blackbox_metrics.txt", "a") as bb_file:
                bb_file.write(f"{epoch}\t{blackbox_loss:.6f}\t{blackbox_acc:.6f}\t{blackbox_precision:.6f}\t{blackbox_recall:.6f}\t{blackbox_f1:.6f}\n")

    # Salvataggio dei modelli
    generator.save("generator_trained.h5")
    substitute_detector.save("substitute_detector_trained.h5")



pretrained_model_path = '../CNN/3/models/malware_classification_model_best.h5'
blackbox_model = load_model(pretrained_model_path)
blackbox_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

image_directory = '../../Images/16x16_grayscale_images'
SIZE = 16
categories = ["Adware", "Backdoor", "Downloader", "Ransomware", "Spyware", "Trojan", "Virus"]
label_dict = {category: i for i, category in enumerate(categories)}
num_classes = len(categories)
latent_dim = 256

dataset = []
labels = []
image_names = []
for category in categories:
    path = os.path.join(image_directory, category)
    img_files = [f for f in os.listdir(path) if f.endswith('.png')]
    for img_name in img_files:
        img_path = os.path.join(path, img_name)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is not None and image.shape == (SIZE, SIZE):
            dataset.append(np.array(image))
            labels.append(label_dict[category])
            image_names.append((img_name, category))

indices = np.arange(len(dataset))
np.random.shuffle(indices)
dataset = np.array(dataset)[indices]
labels = to_categorical(np.array(labels)[indices], num_classes=num_classes)
image_names = np.array(image_names)[indices]

X_train, X_temp, y_train, y_temp = train_test_split(dataset, labels, test_size=0.20, random_state=0)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=0)
X_train = normalize(X_train, axis=1)
X_val = normalize(X_val, axis=1)
X_test = normalize(X_test, axis=1)

substitute_detector = build_substitute_detector((16, 16, 1), num_classes)
generator = build_generator(latent_dim)
generator.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy')
substitute_detector.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

train_models(generator, substitute_detector, blackbox_model, X_train, y_train, X_val, y_val, latent_dim, num_classes)
