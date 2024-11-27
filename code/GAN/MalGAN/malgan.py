import os
import cv2
import numpy as np
import tensorflow as tf
import shap
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical, normalize
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def build_generator(latent_dim, num_classes, image_shape):
    noise = Input(shape=(latent_dim,))
    labels = Input(shape=(num_classes,))
    real_image = Input(shape=image_shape)

    x = Dense(256, activation="relu")(noise)
    x = Dense(np.prod(image_shape), activation="tanh")(x)
    x = Reshape(image_shape)(x)

    noisy_image = Add()([real_image, x])  # Somma del rumore all'immagine reale
    model = Model(inputs=[noise, labels, real_image], outputs=noisy_image, name="generator")
    return model

def build_substitute_detector(input_shape, num_classes):
    input_img = Input(shape=input_shape)
    x = Flatten()(input_img)
    x = Dense(128, activation="relu")(x)
    x = Dense(num_classes, activation="softmax")(x)
    model = Model(inputs=input_img, outputs=x, name="substitute_detector")
    return model

def train_models(generator, substitute_detector, blackbox_model, X_train, y_train, X_val, y_val, latent_dim, num_classes, epochs=100, batch_size=32, beta_param=0.05):
    ensure_dir("models/generator")
    ensure_dir("models/substitute_detector")
    ensure_dir("metrics")
    ensure_dir("generated_images")

    generator_optimizer = Adam(learning_rate=0.001)
    substitute_optimizer = Adam(learning_rate=0.001)

    num_batches = int(np.ceil(len(X_train) / batch_size))

    # Liste per memorizzare metriche
    blackbox_metrics = {
        "accuracy": [],
        "loss": [],
        "f1": [],
        "precision": [],
        "recall": []
    }
    substitute_metrics = {
        "accuracy": [],
        "loss": [],
        "f1": [],
        "precision": [],
        "recall": []
    }
    generator_losses = []
    noise_levels = []

    for epoch in range(epochs):
        print(f"Starting epoch {epoch}")
        epoch_generator_loss = []

        for i in range(num_batches):
            batch_start = i * batch_size
            batch_end = min(batch_start + batch_size, len(X_train))

            noise = np.random.normal(0, 0.25, (batch_end - batch_start, latent_dim)) 
            fake_labels = to_categorical(np.random.randint(0, num_classes, batch_end - batch_start), num_classes)
            real_images = X_train[batch_start:batch_end]
            real_labels = y_train[batch_start:batch_end]

            with tf.GradientTape(persistent=True) as tape:
                fake_images = generator([noise, fake_labels, real_images], training=True)
                fake_predictions = substitute_detector(fake_images, training=False)
                real_predictions = substitute_detector(real_images, training=True)
                blackbox_predictions = blackbox_model(fake_images, training=False)

                generator_loss = (
                    -tf.reduce_mean(tf.math.log(tf.reduce_max(blackbox_predictions, axis=1) + 1e-8)) +  # Ingannare il black-box
                    beta_param * tf.reduce_mean(tf.norm(fake_images - real_images, ord=2))  # Regularizzazione del rumore
                )

                substitute_loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(real_labels, real_predictions))

            gen_gradients = tape.gradient(generator_loss, generator.trainable_variables)
            generator_optimizer.apply_gradients(zip(gen_gradients, generator.trainable_variables))
            sub_gradients = tape.gradient(substitute_loss, substitute_detector.trainable_variables)
            substitute_optimizer.apply_gradients(zip(sub_gradients, substitute_detector.trainable_variables))

            # Salva la loss del generatore per ogni batch
            epoch_generator_loss.append(generator_loss.numpy())

        avg_generator_loss = np.mean(epoch_generator_loss)
        generator_losses.append(avg_generator_loss)

        # Calcola il livello medio di rumore e aggiorna la lista
        average_noise_level = np.mean([np.linalg.norm(fake - real) for fake, real in zip(fake_images, real_images)])
        noise_levels.append(average_noise_level)

        # Validation e calcolo metriche
        val_fake_images = generator.predict([
            np.random.normal(0, 0.25, (len(X_val), latent_dim)),
            to_categorical(np.random.randint(0, num_classes, len(X_val)), num_classes),
            X_val
        ])
        val_blackbox_predictions = blackbox_model.predict(val_fake_images, verbose=0)
        val_real_predictions = substitute_detector.predict(X_val, verbose=0)

        blackbox_acc = accuracy_score(np.argmax(y_val, axis=1), np.argmax(val_blackbox_predictions, axis=1))
        substitute_acc = accuracy_score(np.argmax(y_val, axis=1), np.argmax(val_real_predictions, axis=1))


        # Calcolo di altre metriche
        blackbox_loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_val, val_blackbox_predictions)).numpy()
        substitute_loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_val, val_real_predictions)).numpy()


        blackbox_f1, blackbox_precision, blackbox_recall, _ = precision_recall_fscore_support(np.argmax(y_val, axis=1), np.argmax(val_blackbox_predictions, axis=1), average="weighted")
        substitute_f1, substitute_precision, substitute_recall, _ = precision_recall_fscore_support(np.argmax(y_val, axis=1), np.argmax(val_real_predictions, axis=1), average="weighted")

        blackbox_metrics["accuracy"].append(blackbox_acc)
        blackbox_metrics["loss"].append(blackbox_loss)
        blackbox_metrics["f1"].append(blackbox_f1)
        blackbox_metrics["precision"].append(blackbox_precision)
        blackbox_metrics["recall"].append(blackbox_recall)

        substitute_metrics["accuracy"].append(substitute_acc)
        substitute_metrics["loss"].append(substitute_loss)
        substitute_metrics["f1"].append(substitute_f1)
        substitute_metrics["precision"].append(substitute_precision)
        substitute_metrics["recall"].append(substitute_recall)

        print(f"Epoch {epoch} - Generator Loss: {avg_generator_loss}, Noise Level: {average_noise_level}, Blackbox Accuracy: {blackbox_acc}, Substitute Accuracy: {substitute_acc}")

        # Salva griglia immagini reali e generate
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        for idx, ax in enumerate(axes.flat):
            if idx < 5:
                ax.imshow(X_val[idx].reshape(SIZE, SIZE), cmap='gray')
                ax.set_title("Real")
            else:
                ax.imshow(val_fake_images[idx - 5].reshape(SIZE, SIZE), cmap='gray')
                ax.set_title("Generated")
            ax.axis("off")
        plt.tight_layout()
        plt.savefig(f"generated_images/grid_epoch_{epoch}.png")
        plt.close()
        
        # Salva le metriche
        with open(f"metrics/evaluation_metrics_epoch_{epoch}.txt", "w") as f:
            f.write(f"Epoch {epoch} - Generator Loss: {avg_generator_loss}\n")
            f.write(f"Epoch {epoch} - Noise Level: {average_noise_level}\n")
            f.write(f"Epoch {epoch} - Blackbox Accuracy: {blackbox_acc}, Blackbox Loss: {blackbox_loss}, Blackbox Precision: {blackbox_precision}, Blackbox Recall: {blackbox_recall}, Blackbox F1: {blackbox_f1}\n")
            f.write(f"Epoch {epoch} - Substitute Accuracy: {substitute_acc}, Substitute Loss: {substitute_loss}, Substitute Precision: {substitute_precision}, Substitute Recall: {substitute_recall}, Substitute F1: {substitute_f1}\n")

    # Generazione grafico migliorato
    plt.figure()
    plt.plot(range(epochs), generator_losses, label="Generator Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Generator Loss vs Epochs")
    plt.legend()
    plt.grid()
    plt.savefig("metrics/generator_loss_vs_epochs.png")
    plt.close()

    plt.figure()
    plt.plot(range(epochs), blackbox_metrics["accuracy"], label="Blackbox Accuracy")
    plt.plot(range(epochs), substitute_metrics["accuracy"], label="Substitute Accuracy")
    plt.plot(range(epochs), noise_levels, label="Noise Level")
    plt.fill_between(range(epochs), np.array(blackbox_metrics["accuracy"]) - 0.02, np.array(blackbox_metrics["accuracy"]) + 0.02, alpha=0.2)
    plt.fill_between(range(epochs), np.array(substitute_metrics["accuracy"]) - 0.02, np.array(substitute_metrics["accuracy"]) + 0.02, alpha=0.2)
    plt.xlabel("Epochs")
    plt.ylabel("Metrics")
    plt.title("Performance and Noise vs Epochs")
    plt.legend()
    plt.grid()
    plt.savefig("metrics/performance_and_noise_vs_epochs.png")
    plt.close()

    generator.save("models/generator/generator_trained.keras")
    substitute_detector.save("models/substitute_detector/substitute_detector_trained.keras")

# Parametri
pretrained_model_path = '../CNN/3/models/malware_classification_model_best.h5'
blackbox_model = load_model(pretrained_model_path)
blackbox_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

image_directory = '../../Images/16x16_grayscale_images'
SIZE = 16
categories = ["Adware", "Backdoor", "Downloader", "Ransomware", "Spyware", "Trojan", "Virus"]
label_dict = {category: i for i, category in enumerate(categories)}
num_classes = len(categories)
latent_dim = 256

# Caricamento dataset
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

indices = np.random.permutation(len(dataset))
dataset = np.array(dataset)[indices]
labels = to_categorical(np.array(labels)[indices], num_classes=num_classes)
image_names = np.array(image_names)[indices]

# Split dataset
X_train, X_temp, y_train, y_temp = train_test_split(dataset, labels, test_size=0.20, random_state=0)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=0)
X_train = normalize(X_train, axis=1)
X_val = normalize(X_val, axis=1)
X_test = normalize(X_test, axis=1)

# Aggiungere dimensione del canale
X_train = np.expand_dims(X_train, axis=-1)
X_val = np.expand_dims(X_val, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

# Costruzione dei modelli
substitute_detector = build_substitute_detector((SIZE, SIZE, 1), num_classes)
generator = build_generator(latent_dim, num_classes, (SIZE, SIZE, 1))
generator.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy')
substitute_detector.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Training
train_models(generator, substitute_detector, blackbox_model, X_train, y_train, X_val, y_val, latent_dim, num_classes)




# import os
# import cv2
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import load_model, Model
# from tensorflow.keras.layers import Input, Dense, Reshape, Flatten
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.utils import to_categorical, normalize
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# def ensure_dir(path):
#     os.makedirs(path, exist_ok=True)

# def build_generator(latent_dim):
#     noise = Input(shape=(latent_dim,))
#     x = Dense(256, activation="relu")(noise)
#     x = Dense(16 * 16, activation="softmax")(x)
#     img = Reshape((16, 16, 1))(x)
#     model = Model(inputs=noise, outputs=img, name="generator")
#     return model

# def build_substitute_detector(input_shape, num_classes):
#     input_img = Input(shape=input_shape)
#     x = Flatten()(input_img)
#     x = Dense(128, activation="relu")(x)
#     x = Dense(64, activation="relu")(x)
#     output = Dense(num_classes, activation="softmax")(x)
#     model = Model(inputs=input_img, outputs=output, name="substitute_detector")
#     return model

# def custom_loss(generator_output, blackbox_pred, substitute_pred):
#     discrepancy_loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(blackbox_pred, substitute_pred))
#     validity_loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(tf.ones_like(blackbox_pred) / num_classes, blackbox_pred))
#     return discrepancy_loss + validity_loss

# def train_models(generator, substitute_detector, blackbox_model, X_train, y_train, X_val, y_val, latent_dim, num_classes, epochs=10, batch_size=32):
#     ensure_dir("blackbox")
#     ensure_dir("generator")
#     ensure_dir("substitute_detector")
    
#     with open("blackbox/blackbox_metrics.txt", "w") as bb_file, \
#          open("generator/generator_metrics.txt", "w") as gen_file, \
#          open("substitute_detector/substitute_metrics.txt", "w") as sub_file:
        
#         bb_file.write("Epoch\tLoss\tAccuracy\tPrecision\tRecall\tF1-Score\n")
#         gen_file.write("Epoch\tLoss\tAccuracy\tPrecision\tRecall\tF1-Score\n")
#         sub_file.write("Epoch\tLoss\tAccuracy\tPrecision\tRecall\tF1-Score\n")

#         for epoch in range(epochs):
#             for _ in range(len(X_train) // batch_size):
#                 idx = np.random.randint(0, X_train.shape[0], batch_size)
#                 real_images = np.expand_dims(X_train[idx], axis=-1)
#                 real_labels = y_train[idx]

#                 substitute_loss, substitute_acc = substitute_detector.train_on_batch(real_images, real_labels)

#                 noise = np.random.normal(0, 1, (batch_size, latent_dim))
#                 with tf.GradientTape() as tape:
#                     fake_images = generator(noise, training=True)
#                     substitute_pred = substitute_detector(fake_images, training=False)
#                     blackbox_pred = blackbox_model(real_images, training=False)
#                     g_loss = custom_loss(fake_images, blackbox_pred, substitute_pred)

#                 gradients = tape.gradient(g_loss, generator.trainable_variables)
#                 generator.optimizer.apply_gradients(zip(gradients, generator.trainable_variables))

#             if epoch % 10 == 0:
#                 val_images = np.expand_dims(X_val, axis=-1)
#                 val_loss, val_acc = substitute_detector.evaluate(val_images, y_val, verbose=0)
#                 val_pred = substitute_detector.predict(val_images)
#                 val_pred_classes = np.argmax(val_pred, axis=1)
#                 val_true_classes = np.argmax(y_val, axis=1)

#                 substitute_precision = precision_score(val_true_classes, val_pred_classes, average="macro")
#                 substitute_recall = recall_score(val_true_classes, val_pred_classes, average="macro")
#                 substitute_f1 = f1_score(val_true_classes, val_pred_classes, average="macro")

#                 bb_loss, bb_acc = blackbox_model.evaluate(val_images, y_val, verbose=0)
#                 bb_pred = blackbox_model.predict(val_images)
#                 bb_pred_classes = np.argmax(bb_pred, axis=1)

#                 blackbox_precision = precision_score(val_true_classes, bb_pred_classes, average="macro")
#                 blackbox_recall = recall_score(val_true_classes, bb_pred_classes, average="macro")
#                 blackbox_f1 = f1_score(val_true_classes, bb_pred_classes, average="macro")

#                 bb_file.write(f"{epoch}\t{bb_loss:.6f}\t{bb_acc:.6f}\t{blackbox_precision:.6f}\t{blackbox_recall:.6f}\t{blackbox_f1:.6f}\n")
#                 gen_file.write(f"{epoch}\t{g_loss.numpy():.6f}\t0\t0\t0\t0\n")
#                 sub_file.write(f"{epoch}\t{val_loss:.6f}\t{val_acc:.6f}\t{substitute_precision:.6f}\t{substitute_recall:.6f}\t{substitute_f1:.6f}\n")

#     generator.save("generator/generator_trained.h5")
#     substitute_detector.save("substitute_detector/substitute_detector_trained.h5")

# pretrained_model_path = '../CNN/3/models/malware_classification_model_best.h5'
# blackbox_model = load_model(pretrained_model_path)
# blackbox_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# image_directory = '../../Images/16x16_grayscale_images'
# SIZE = 16
# categories = ["Adware", "Backdoor", "Downloader", "Ransomware", "Spyware", "Trojan", "Virus"]
# label_dict = {category: i for i, category in enumerate(categories)}
# num_classes = len(categories)
# latent_dim = 256

# dataset = []
# labels = []
# image_names = []
# for category in categories:
#     path = os.path.join(image_directory, category)
#     img_files = [f for f in os.listdir(path) if f.endswith('.png')]
#     for img_name in img_files:
#         img_path = os.path.join(path, img_name)
#         image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
#         if image is not None and image.shape == (SIZE, SIZE):
#             dataset.append(np.array(image))
#             labels.append(label_dict[category])
#             image_names.append((img_name, category))

# indices = np.arange(len(dataset))
# np.random.shuffle(indices)
# dataset = np.array(dataset)[indices]
# labels = to_categorical(np.array(labels)[indices], num_classes=num_classes)
# image_names = np.array(image_names)[indices]

# X_train, X_temp, y_train, y_temp = train_test_split(dataset, labels, test_size=0.20, random_state=0)
# X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=0)
# X_train = normalize(X_train, axis=1)
# X_val = normalize(X_val, axis=1)
# X_test = normalize(X_test, axis=1)

# substitute_detector = build_substitute_detector((16, 16, 1), num_classes)
# generator = build_generator(latent_dim)
# generator.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy')
# substitute_detector.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# train_models(generator, substitute_detector, blackbox_model, X_train, y_train, X_val, y_val, latent_dim, num_classes)



############### CODICE PIÙ CORRETTO ################
# import os
# import cv2
# import numpy as np
# import tensorflow as tf
# import matplotlib.pyplot as plt
# from tensorflow.keras.models import load_model, Model
# from tensorflow.keras.layers import Input, Dense, Reshape, Flatten
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.utils import to_categorical, normalize
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

# def ensure_dir(path):
#     os.makedirs(path, exist_ok=True)

# def build_generator(latent_dim):
#     noise = Input(shape=(latent_dim,))
#     x = Dense(256, activation="relu")(noise)
#     x = Dense(16 * 16, activation="softmax")(x)
#     img = Reshape((16, 16, 1))(x)
#     model = Model(inputs=noise, outputs=img, name="generator")
#     return model

# def build_substitute_detector(input_shape, num_classes):
#     input_img = Input(shape=input_shape)
#     x = Flatten()(input_img)
#     x = Dense(128, activation="relu")(x)
#     x = Dense(64, activation="relu")(x)
#     output = Dense(num_classes, activation="softmax")(x)
#     model = Model(inputs=input_img, outputs=output, name="substitute_detector")
#     # model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
#     return model

# def train_models(generator, substitute_detector, blackbox_model, X_train, y_train, X_val, y_val, latent_dim, num_classes, epochs=100, batch_size=32):
#     # Creazione dei file per salvare le metriche
#     ensure_dir("blackbox")
#     ensure_dir("substitute_detector")
#     ensure_dir("generator")
#     with open("blackbox/blackbox_metrics.txt", "w") as bb_file, open("substitute_detector/substitute_metrics.txt", "w") as sub_file, open("generator/generator_metrics.txt", "w") as gen_file:
#         # Intestazioni per metriche estese
#         bb_file.write("Epoch\tLoss\tAccuracy\tPrecision\tRecall\tF1-Score\n")
#         sub_file.write("Epoch\tLoss\tAccuracy\tPrecision\tRecall\tF1-Score\n")
#         gen_file.write("Epoch\tLoss\tAccuracy\tPrecision\tRecall\tF1-Score\n")

#     # Loop di addestramento
#     for epoch in range(epochs):
#         # 1. Addestramento del substitute detector su dati reali
#         idx = np.random.randint(0, X_train.shape[0], batch_size)
#         real_images = X_train[idx]
#         real_labels = y_train[idx]

#         # Aggiungere dimensione del canale per immagini reali
#         real_images = np.expand_dims(real_images, axis=-1)
#         substitute_loss, substitute_acc = substitute_detector.train_on_batch(real_images, real_labels)

#         # 2. Generazione di immagini false
#         noise = np.random.normal(0, 1, (batch_size, latent_dim))
#         fake_images = generator.predict(noise)

#         # 3. Ottenere previsioni per immagini false dal substitute detector
#         fake_predictions = substitute_detector.predict(fake_images)

#         # 4. Calcolo della perdita per il generatore
#         # Obiettivo: fare in modo che il substitute detector classifichi le immagini false come una distribuzione uniforme
#         uniform_labels = np.ones_like(fake_predictions) / num_classes  # Distribuzione uniforme
#         generator_loss = tf.reduce_mean(
#             tf.keras.losses.categorical_crossentropy(uniform_labels, fake_predictions)
#         )

#         # Calcolo delle metriche per il generatore
#         gen_pred_classes = np.argmax(fake_predictions, axis=1)
#         gen_target_classes = np.argmax(uniform_labels, axis=1)

#         gen_accuracy = accuracy_score(gen_target_classes, gen_pred_classes)
#         gen_precision = precision_score(gen_target_classes, gen_pred_classes, average="macro")
#         gen_recall = recall_score(gen_target_classes, gen_pred_classes, average="macro")
#         gen_f1 = f1_score(gen_target_classes, gen_pred_classes, average="macro")

#         # 5. Aggiornamento del generatore
#         with tf.GradientTape() as tape:
#             fake_images = generator(noise)  # Ricrea le immagini false
#             fake_predictions = substitute_detector(fake_images)  # Previsioni sulle immagini false
#             generator_loss = tf.reduce_mean(
#                 tf.keras.losses.categorical_crossentropy(uniform_labels, fake_predictions)
#             )
#         gradients = tape.gradient(generator_loss, generator.trainable_variables)
#         generator.optimizer.apply_gradients(zip(gradients, generator.trainable_variables))

#         # 6. Validazione e salvataggio delle metriche ogni 10 epoche
#         if epoch % 10 == 0:
#             # Validazione del substitute detector
#             val_images = np.expand_dims(X_val, axis=-1)
#             val_loss, val_acc = substitute_detector.evaluate(val_images, y_val, verbose=0)

#             # Calcolo delle previsioni e delle metriche dettagliate
#             val_predictions = substitute_detector.predict(val_images)
#             val_pred_classes = np.argmax(val_predictions, axis=1)
#             val_true_classes = np.argmax(y_val, axis=1)

#             substitute_precision = precision_score(val_true_classes, val_pred_classes, average="macro")
#             substitute_recall = recall_score(val_true_classes, val_pred_classes, average="macro")
#             substitute_f1 = f1_score(val_true_classes, val_pred_classes, average="macro")

#             # Validazione del blackbox model
#             blackbox_loss, blackbox_acc = blackbox_model.evaluate(val_images, y_val, verbose=0)
#             blackbox_predictions = blackbox_model.predict(val_images)
#             blackbox_pred_classes = np.argmax(blackbox_predictions, axis=1)

#             blackbox_precision = precision_score(val_true_classes, blackbox_pred_classes, average="macro")
#             blackbox_recall = recall_score(val_true_classes, blackbox_pred_classes, average="macro")
#             blackbox_f1 = f1_score(val_true_classes, blackbox_pred_classes, average="macro")

#             # Log dei risultati
#             print(f"Epoch {epoch}: Substitute Detector -> Loss: {val_loss}, Accuracy: {val_acc}")
#             print(f"Epoch {epoch}: Blackbox Model -> Loss: {blackbox_loss}, Accuracy: {blackbox_acc}")
#             print(f"Generator Loss: {generator_loss.numpy()}, Accuracy: {gen_accuracy}, Precision: {gen_precision}, Recall: {gen_recall}, F1-Score: {gen_f1}")

#             # Salvataggio delle metriche estese nei file
#             with open("substitute_detector/substitute_metrics.txt", "a") as sub_file:
#                 sub_file.write(f"{epoch}\t{val_loss:.6f}\t{val_acc:.6f}\t{substitute_precision:.6f}\t{substitute_recall:.6f}\t{substitute_f1:.6f}\n")

#             with open("blackbox/blackbox_metrics.txt", "a") as bb_file:
#                 bb_file.write(f"{epoch}\t{blackbox_loss:.6f}\t{blackbox_acc:.6f}\t{blackbox_precision:.6f}\t{blackbox_recall:.6f}\t{blackbox_f1:.6f}\n")

#             with open("generator/generator_metrics.txt", "a") as gen_file:
#                 gen_file.write(f"{epoch}\t{generator_loss.numpy():.6f}\t{gen_accuracy:.6f}\t{gen_precision:.6f}\t{gen_recall:.6f}\t{gen_f1:.6f}\n")

#     # Salvataggio dei modelli
#     generator.save("generator/generator_trained.h5")
#     substitute_detector.save("substitute_detector/substitute_detector_trained.h5")

# pretrained_model_path = '../CNN/3/models/malware_classification_model_best.h5'
# blackbox_model = load_model(pretrained_model_path)
# blackbox_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# image_directory = '../../Images/16x16_grayscale_images'
# SIZE = 16
# categories = ["Adware", "Backdoor", "Downloader", "Ransomware", "Spyware", "Trojan", "Virus"]
# label_dict = {category: i for i, category in enumerate(categories)}
# num_classes = len(categories)
# latent_dim = 256

# dataset = []
# labels = []
# image_names = []
# for category in categories:
#     path = os.path.join(image_directory, category)
#     img_files = [f for f in os.listdir(path) if f.endswith('.png')]
#     for img_name in img_files:
#         img_path = os.path.join(path, img_name)
#         image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
#         if image is not None and image.shape == (SIZE, SIZE):
#             dataset.append(np.array(image))
#             labels.append(label_dict[category])
#             image_names.append((img_name, category))

# indices = np.arange(len(dataset))
# np.random.shuffle(indices)
# dataset = np.array(dataset)[indices]
# labels = to_categorical(np.array(labels)[indices], num_classes=num_classes)
# image_names = np.array(image_names)[indices]

# X_train, X_temp, y_train, y_temp = train_test_split(dataset, labels, test_size=0.20, random_state=0)
# X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=0)
# X_train = normalize(X_train, axis=1)
# X_val = normalize(X_val, axis=1)
# X_test = normalize(X_test, axis=1)

# substitute_detector = build_substitute_detector((16, 16, 1), num_classes)
# generator = build_generator(latent_dim)
# generator.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy')
# substitute_detector.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# train_models(generator, substitute_detector, blackbox_model, X_train, y_train, X_val, y_val, latent_dim, num_classes)
