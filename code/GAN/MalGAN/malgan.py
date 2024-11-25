import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical, normalize
from sklearn.model_selection import train_test_split
import pickle
from concurrent.futures import ThreadPoolExecutor
from sklearn.metrics import f1_score
import seaborn as sns

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def log_message(file_path, accuracy, noise_level, precision, f1, fooled):
    with open(file_path, "a") as log_file:
        log_file.write(f"Iterazione: {len(accuracy)-1}, Accuratezza della predizione: {accuracy[-1]*100:.2f}%, "
                       f"Livello di rumore nell'immagine: {noise_level*100:.2f}%, "
                       f"Precisazione della predizione: {precision*100:.2f}%, "
                       f"F1-Score: {f1:.2f}, Modello ingannato: {'si' if not fooled else 'no'}\n")


def collect_graph_data(accuracy, graph_dir, img_name):
    ensure_dir(graph_dir)
    with open(os.path.join(graph_dir, f"{img_name}_data.pkl"), "wb") as f:
        pickle.dump(accuracy, f)

def setup_directories():
    base_dirs = ['blackbox', 'discriminator']
    for base_dir in base_dirs:
        for category in categories:
            for subdir in ["images", "train_log", "graphs"]:
                path = os.path.join(base_dir, category, subdir)
                ensure_dir(path)

def generate_and_save_graphs():
    for model_type in ['blackbox', 'discriminator']:
        for category in categories:
            graph_data_path = os.path.join(model_type, category, "graphs")
            graph_files = os.listdir(graph_data_path)
            all_accuracies = []
            max_length = 0

            # Carica i dati di accuratezza da ogni file
            for data_file in graph_files:
                with open(os.path.join(graph_data_path, data_file), "rb") as f:
                    accuracies = pickle.load(f)
                    if len(accuracies) > max_length:
                        max_length = len(accuracies)
                    all_accuracies.append(accuracies)

            if all_accuracies:
                # Normalizza la lunghezza delle serie
                all_accuracies = np.array([acc + [np.nan] * (max_length - len(acc)) for acc in all_accuracies], dtype=float)

                # Calcola la media e la deviazione standard delle accuratezze per ogni iterazione
                mean_accuracies = np.nanmean(all_accuracies, axis=0)
                std_accuracies = np.nanstd(all_accuracies, axis=0)

                # Grafico a barre con errore
                plt.figure(figsize=(12, 8))
                plt.bar(range(max_length), mean_accuracies, yerr=std_accuracies, capsize=5, color='skyblue', alpha=0.7)
                plt.title(f'Mean Accuracy Trend for {category} - {model_type}')
                plt.xlabel('Iterations')
                plt.ylabel('Mean Accuracy')
                plt.grid(True, linestyle='--', alpha=0.6)
                plt.savefig(os.path.join(graph_data_path, f"{category}_mean_accuracy_bar_{model_type}.png"))
                plt.close()

                # Heatmap
                plt.figure(figsize=(12, 8))
                sns.heatmap(all_accuracies, annot=False, cmap='coolwarm', cbar_kws={'label': 'Accuracy'})
                plt.title(f'Heatmap of Accuracy Trends for {category} - {model_type}')
                plt.xlabel('Iterations')
                plt.ylabel('Test Instance')
                plt.savefig(os.path.join(graph_data_path, f"{category}_accuracy_heatmap_{model_type}.png"))
                plt.close()

               # Grafico a linee con media mobile per lisciare le variazioni
                plt.figure(figsize=(12, 8))
                smoothed_accuracies = np.convolve(mean_accuracies, np.ones(3)/3, mode='same')  # Media mobile con finestra di 3
                plt.plot(range(max_length), smoothed_accuracies, label='Smoothed Mean Accuracy', color='blue', linewidth=2)
                plt.title(f'Smoothed Accuracy Trend for {category} - {model_type}')
                plt.xlabel('Iterations')
                plt.ylabel('Smoothed Mean Accuracy')
                plt.grid(True, linestyle='--', alpha=0.6)
                plt.legend()
                plt.savefig(os.path.join(graph_data_path, f"{category}_smoothed_accuracy_trend_{model_type}.png"))
                plt.close()

                # Elimina i file di dati originali dopo aver creato i grafici
                for file in graph_files:
                    os.remove(os.path.join(graph_data_path, file))

                print(f"All data files in {graph_data_path} have been deleted after generating the graphs.")



pretrained_model_path = '../CNN/3/models/malware_classification_model_best.h5'
blackbox_model = load_model(pretrained_model_path)
blackbox_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

def build_generator(latent_dim):
    noise = Input(shape=(latent_dim,))
    x = Dense(256, activation="relu")(noise)
    img = Reshape((16, 16, 1))(x)
    return Model(inputs=noise, outputs=img, name="generator")

def build_discriminator(input_shape):
    input_img = Input(shape=input_shape)
    x = Flatten()(input_img)
    x = Dense(128, activation="relu")(x)
    x = Dense(64, activation="relu")(x)
    output = Dense(1, activation="sigmoid")(x)
    return Model(inputs=input_img, outputs=output, name="discriminator")

latent_dim = 256
generator = build_generator(latent_dim)
discriminator = build_discriminator((16, 16, 1))
discriminator.compile(optimizer=Adam(0.002), loss='binary_crossentropy', metrics=['accuracy'])

image_directory = '../../Images/16x16_grayscale_images'
SIZE = 16
categories = ["Adware", "Backdoor", "Downloader", "Ransomware", "Spyware", "Trojan", "Virus"]
label_dict = {category: i for i, category in enumerate(categories)}

dataset = []
labels = []
image_names = []

for category in categories:
    path = os.path.join(image_directory, category)
    img_files = [f for f in os.listdir(path) if f.endswith('.png')]
    print(f"Loaded {len(img_files)} images from category {category}")
    for img_name in img_files:
        img_path = os.path.join(path, img_name)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is not None and image.shape == (SIZE, SIZE):
            dataset.append(np.array(image))
            labels.append(label_dict[category])
            image_names.append((img_name, category))

dataset = np.array(dataset).reshape(-1, SIZE, SIZE, 1)
labels = to_categorical(np.array(labels), num_classes=len(categories))

X_train, X_temp, y_train, y_temp = train_test_split(dataset, labels, test_size=0.20, random_state=20)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=20)

X_train = normalize(X_train, axis=1)
X_val = normalize(X_val, axis=1)
X_test = normalize(X_test, axis=1)

setup_directories()

def test_and_manipulate_image(index, img, label_index, category, model_type, img_name):
    img_dir = os.path.join(model_type, category, "images")
    log_dir = os.path.join(model_type, category, "train_log")
    graph_dir = os.path.join(model_type, category, "graphs")

    img = np.expand_dims(img, axis=0)
    predictions = blackbox_model.predict(img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    initial_accuracy = predictions[0][predicted_class_index]
    is_correct = 1 if predicted_class_index == label_index else 0

    noise = np.random.normal(0, 0.1, (1, latent_dim))
    accuracies = [is_correct]
    base_noise_level = 0.1
    noise_increase_factor = 0.05  # Incremento lineare

    for iteration in range(10):
        noise += np.random.normal(0, base_noise_level + iteration * noise_increase_factor, noise.shape)
        generated_image = generator.predict(noise)
        preds = blackbox_model.predict(generated_image)
        new_accuracy = preds[0][np.argmax(preds, axis=1)[0]]
        new_is_correct = 1 if np.argmax(preds, axis=1)[0] == label_index else 0
        accuracies.append(new_is_correct)

        true_labels = [label_index]  # La vera classe
        predicted_labels = [np.argmax(preds, axis=1)[0]]  # Classe predetta
        f1 = f1_score(true_labels, predicted_labels, average='macro')

        log_message(os.path.join(log_dir, f"{img_name}_log.txt"), accuracies, noise_increase_factor * (iteration + 1), new_accuracy, new_is_correct, f1)


        if not new_is_correct:
            manipulated_img_path = os.path.join(img_dir, f"{img_name}_fooled_at_iter_{iteration+1}.png")
            cv2.imwrite(manipulated_img_path, generated_image[0, :, :, 0] * 255)
            break

    collect_graph_data(accuracies, graph_dir, img_name)

def process_images():
    print(f"Total images in test set: {len(X_test)}")
    category_count = {cat: 0 for cat in categories}

    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = []
        for i, (img, (img_name, category)) in enumerate(zip(X_test, image_names)):
            category_count[category] += 1
            print(f"Processing {category}, Image: {img_name}")
            future = executor.submit(test_and_manipulate_image, i, img, np.argmax(y_test[i]), category, 'blackbox', img_name)
            futures.append(future)
            future = executor.submit(test_and_manipulate_image, i, img, np.argmax(y_test[i]), category, 'discriminator', img_name)
            futures.append(future)
        for future in futures:
            future.result()

    for cat, count in category_count.items():
        print(f"Processed {count} images from category {cat}")
    generate_and_save_graphs()

process_images()





# import os
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# from tensorflow.keras.models import load_model, Model
# from tensorflow.keras.layers import Input, Dense, Reshape, Flatten
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.utils import to_categorical, normalize
# from sklearn.model_selection import train_test_split
# import pickle
# from concurrent.futures import ThreadPoolExecutor

# def ensure_dir(path):
#     os.makedirs(path, exist_ok=True)

# def log_message(file_path, message):
#     with open(file_path, "a") as log_file:
#         log_file.write(f"{message}\n")

# def collect_graph_data(accuracy, graph_dir, img_name):
#     ensure_dir(graph_dir)
#     with open(os.path.join(graph_dir, f"{img_name}_data.pkl"), "wb") as f:
#         pickle.dump(accuracy, f)

# def setup_directories():
#     base_dirs = ['blackbox', 'discriminator']
#     for base_dir in base_dirs:
#         for category in categories:
#             for subdir in ["images", "train_log", "graphs"]:
#                 path = os.path.join(base_dir, category, subdir)
#                 ensure_dir(path)

# def generate_and_save_graphs():
#     for model_type in ['blackbox', 'discriminator']:
#         for category in categories:
#             graph_data_path = os.path.join(model_type, category, "graphs")
#             if not os.listdir(graph_data_path):
#                 print(f"No data files found in {graph_data_path} for category {category}. No graphs generated.")
#                 continue
#             for data_file in os.listdir(graph_data_path):
#                 img_name = data_file.split('_')[0]
#                 with open(os.path.join(graph_data_path, data_file), "rb") as f:
#                     accuracies = pickle.load(f)
#                     save_graph(accuracies, os.path.join(graph_data_path, f"{img_name}_accuracy.png"), img_name)

# def save_graph(accuracy, graph_path, name):
#     plt.figure()
#     plt.plot(range(len(accuracy)), accuracy, marker='o')
#     plt.title(f'Accuracy per Iteration for Image {name}')
#     plt.xlabel('Iterations')
#     plt.ylabel('Accuracy')
#     plt.grid(True)
#     plt.savefig(graph_path)
#     plt.close()

# pretrained_model_path = '../CNN/3/models/malware_classification_model_best.h5'
# blackbox_model = load_model(pretrained_model_path)
# blackbox_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# def build_generator(latent_dim):
#     noise = Input(shape=(latent_dim,))
#     x = Dense(256, activation="relu")(noise)
#     img = Reshape((16, 16, 1))(x)
#     return Model(inputs=noise, outputs=img, name="generator")

# def build_discriminator(input_shape):
#     input_img = Input(shape=input_shape)
#     x = Flatten()(input_img)
#     x = Dense(128, activation="relu")(x)
#     x = Dense(64, activation="relu")(x)
#     output = Dense(1, activation="sigmoid")(x)
#     return Model(inputs=input_img, outputs=output, name="discriminator")

# latent_dim = 256
# generator = build_generator(latent_dim)
# discriminator = build_discriminator((16, 16, 1))
# discriminator.compile(optimizer=Adam(0.002), loss='binary_crossentropy', metrics=['accuracy'])

# image_directory = '../../Images/16x16_grayscale_images'
# SIZE = 16
# categories = ["Adware", "Backdoor", "Downloader", "Ransomware", "Spyware", "Trojan", "Virus"]
# label_dict = {category: i for i, category in enumerate(categories)}

# dataset = []
# labels = []
# image_names = []

# for category in categories:
#     path = os.path.join(image_directory, category)
#     img_files = [f for f in os.listdir(path) if f.endswith('.png')]
#     print(f"Loaded {len(img_files)} images from category {category}")
#     for img_name in img_files:
#         img_path = os.path.join(path, img_name)
#         image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
#         if image is not None and image.shape == (SIZE, SIZE):
#             dataset.append(np.array(image))
#             labels.append(label_dict[category])
#             image_names.append((img_name, category))

# dataset = np.array(dataset).reshape(-1, SIZE, SIZE, 1)
# labels = to_categorical(np.array(labels), num_classes=len(categories))

# X_train, X_temp, y_train, y_temp = train_test_split(dataset, labels, test_size=0.30, random_state=0)
# X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=0)

# X_train = normalize(X_train, axis=1)
# X_val = normalize(X_val, axis=1)
# X_test = normalize(X_test, axis=1)

# setup_directories()

# def test_and_manipulate_image(index, img, label_index, category, model_type, img_name):
#     img_dir = os.path.join(model_type, category, "images")
#     log_dir = os.path.join(model_type, category, "train_log")
#     graph_dir = os.path.join(model_type, category, "graphs")

#     img = np.expand_dims(img, axis=0)
#     predictions = blackbox_model.predict(img)
#     predicted_class_index = np.argmax(predictions, axis=1)[0]
#     is_correct = 1 if predicted_class_index == label_index else 0

#     noise = np.random.normal(0, 0.1, (1, latent_dim))
#     accuracies = [is_correct]

#     for iteration in range(10):
#         noise_increase = np.linalg.norm(noise) * 0.1
#         noise += np.random.normal(0, noise_increase, noise.shape)
#         generated_image = generator.predict(noise)
#         preds = blackbox_model.predict(generated_image)
#         new_is_correct = 1 if np.argmax(preds, axis=1)[0] == label_index else 0
#         accuracies.append(new_is_correct)

#         if not new_is_correct:
#             manipulated_img_path = os.path.join(img_dir, f"{img_name}_fooled_at_iter_{iteration+1}.png")
#             cv2.imwrite(manipulated_img_path, generated_image[0, :, :, 0] * 255)
#             log_message(os.path.join(log_dir, f"{img_name}_log.txt"), f"{model_type} fooled at iteration {iteration+1}")
#             break
#         else:
#             log_message(os.path.join(log_dir, f"{img_name}_log.txt"), f"{model_type} not fooled at iteration {iteration+1}")

#     collect_graph_data(accuracies, graph_dir, img_name)

# def process_images():
#     print(f"Total images in test set: {len(X_test)}")
#     category_count = {cat: 0 for cat in categories}

#     with ThreadPoolExecutor(max_workers=16) as executor:
#         futures = []
#         for i, (img, (img_name, category)) in enumerate(zip(X_test, image_names)):
#             category_count[category] += 1
#             print(f"Processing {category}, Image: {img_name}")
#             future = executor.submit(test_and_manipulate_image, i, img, np.argmax(y_test[i]), category, 'blackbox', img_name)
#             futures.append(future)
#             future = executor.submit(test_and_manipulate_image, i, img, np.argmax(y_test[i]), category, 'discriminator', img_name)
#             futures.append(future)
#         for future in futures:
#             future.result()

#     for cat, count in category_count.items():
#         print(f"Processed {count} images from category {cat}")
#     generate_and_save_graphs()

# process_images()

















# import os
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from tensorflow.keras.models import Model, load_model
# from tensorflow.keras.layers import Input, Dense, Reshape, BatchNormalization
# from tensorflow.keras.optimizers import Adam
# import tensorflow as tf
# from datetime import datetime
# import cv2
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report, confusion_matrix

# sns.set(style="white")
# log_file_path = "training_log.txt"

# # Save the log messages to a file
# def log_message(message):
#     timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#     with open(log_file_path, "w") as log_file:
#         log_file.write(f"{timestamp} - {message}\n")

# # Path to blackbox (pretrained model)
# pretrained_model_path = '../CNN/3/models/malware_classification_model.h5'
# blackbox_model = load_model(pretrained_model_path)
# blackbox_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# blackbox_model.trainable = False

# # Generator
# def build_generator(img_shape, latent_dim):
#     noise = Input(shape=(latent_dim,))
#     x = Dense(128 * 4 * 4, activation="relu")(noise)
#     x = BatchNormalization(momentum=0.8)(x)
#     x = Reshape((4, 4, 128))(x)
#     x = tf.keras.layers.Conv2DTranspose(64, kernel_size=3, strides=2, padding="same", activation="relu")(x)
#     x = BatchNormalization(momentum=0.8)(x)
#     x = tf.keras.layers.Conv2DTranspose(1, kernel_size=3, strides=2, padding="same", activation="tanh")(x)
#     img = Reshape(img_shape)(x)
#     return Model(noise, img, name="generator")

# # Discriminator
# def build_substitute_detector(num_classes):
#     inputs = Input(shape=(16, 16, 1))
#     # Aumenta il numero di filtri e aggiungi più strati convoluzionali
#     x = tf.keras.layers.Conv2D(64, kernel_size=3, strides=1, padding="same", activation="relu")(inputs)
#     # x = tf.keras.layers.BatchNormalization()(x)  # Aggiungi Batch Normalization
#     x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    
#     x = tf.keras.layers.Conv2D(64, kernel_size=3, strides=1, padding="same", activation="relu")(x)
#     x = tf.keras.layers.BatchNormalization()(x)
#     x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    
#     x = tf.keras.layers.Conv2D(32, kernel_size=3, strides=1, padding="same", activation="relu")(x)
#     x = tf.keras.layers.BatchNormalization()(x)
#     x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)

#     x = tf.keras.layers.Flatten()(x)
    
#     x = Dense(256, activation="relu")(x)
#     x = tf.keras.layers.Dropout(0.5)(x)  
#     x = Dense(128, activation="relu")(x)
#     x = tf.keras.layers.Dropout(0.5)(x)
    
#     outputs = Dense(num_classes, activation="softmax")(x)
#     return Model(inputs, outputs, name="substitute_detector")


# # Configurazione generale
# latent_dim = 10
# num_classes = 7  # Numero di classi
# SIZE = 16

# # Generatore e rilevatore sostituto
# generator = build_generator((16, 16, 1), latent_dim)
# substitute_detector = build_substitute_detector(num_classes)
# substitute_detector.compile(optimizer=Adam(0.002), loss='categorical_crossentropy', metrics=['accuracy'])
# generator_optimizer = Adam(0.002)

# # Function for training the generator, 
# # Create false images
# # Evaluate the images using the Substitute Detector.
# # Calculates the loss between the desired predictions (Blackbox) and the actual predictions.
# #  Update generator weights to improve images.
# @tf.function(reduce_retracing=True)
# def train_generator_step(generator, substitute_detector, noise, blackbox_preds, optimizer):
#     with tf.GradientTape() as tape:
#         gen_imgs = generator(noise, training=True)
#         fake_preds = substitute_detector(gen_imgs, training=False)
#         loss = tf.keras.losses.categorical_crossentropy(blackbox_preds, fake_preds)
#     grads = tape.gradient(loss, generator.trainable_weights)
#     optimizer.apply_gradients(zip(grads, generator.trainable_weights))
#     return loss


# # Evaluate function:
# # 1. Generator: generate the images and save them
# # 2. Evaluate the Substitute Detector:
# #   - Generate detailed reports and save them in the log files
# def evaluate_and_report(model, test_imgs, test_labels, generator=None, blackbox=None):
#     if generator:
#         noise = np.random.normal(0, 1, (10, latent_dim))
#         generated_imgs = generator.predict(noise)
#         for i, img in enumerate(generated_imgs):
#             plt.figure()
#             plt.imshow(img.reshape(16, 16), cmap="gray")
#             plt.axis("off")
#             plt.savefig(f"./images/generated_image_{i}.png")
#             plt.close()
    
#     preds = model.predict(test_imgs)
#     y_true = np.argmax(test_labels, axis=1)
#     y_pred = np.argmax(preds, axis=1)
#     report = classification_report(y_true, y_pred, target_names=categories, zero_division=1)
#     log_message("\nSubstitute Detector Report:\n" + report)
#     print("\nSubstitute Detector Report:\n", report)

#     # Blackbox's evaluation
#     if blackbox:
#         blackbox_preds = blackbox.predict(test_imgs)
#         y_pred_blackbox = np.argmax(blackbox_preds, axis=1)
#         report_blackbox = classification_report(y_true, y_pred_blackbox, target_names=categories, zero_division=1)
#         log_message("\nBlackbox Report:\n" + report_blackbox)
#         print("\nBlackbox Report:\n", report_blackbox)

# # Data upload and preprocessing
# image_directory = '16x16_grayscale_images/'
# dataset, labels = [], []
# categories = ['Adware', 'Backdoor', 'Downloader', 'Ransomware', 'Spyware', 'Trojan', 'Virus']
# label_dict = {category: i for i, category in enumerate(categories)}

# for category in categories:
#     path = os.path.join(image_directory, category)
#     for img in os.listdir(path):
#         img_path = os.path.join(path, img)
#         image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
#         if image is not None and image.shape == (SIZE, SIZE):
#             dataset.append(image)
#             labels.append(label_dict[category])
#         else:
#             log_message(f"Invalid image skipped: {img_path}")

# dataset = np.array(dataset).reshape(-1, SIZE, SIZE, 1)
# dataset = (dataset.astype(np.float32) - 127.5) / 127.5
# labels = tf.keras.utils.to_categorical(labels, num_classes)

# # Dataset split
# train_imgs, test_imgs, train_labels, test_labels = train_test_split(
#     dataset, labels, test_size=0.2, random_state=10, stratify=labels
# )
# val_imgs, test_imgs, val_labels, test_labels = train_test_split(
#     test_imgs, test_labels, test_size=0.5, random_state=10
# )

# log_message(f"Dataset suddiviso: {train_imgs.shape[0]} train, {val_imgs.shape[0]} val, {test_imgs.shape[0]} test")

# # Training function
# # Training function
# def train(epochs, batch_size=32, save_interval=50):
#     d_losses, g_losses, acc_list, generator_success, pretrained_losses, pretrained_accs = [], [], [], [], [], []

#     for epoch in range(epochs):
#         idx = np.random.randint(0, train_imgs.shape[0], batch_size)
#         real_imgs = tf.convert_to_tensor(train_imgs[idx], dtype=tf.float32)
#         real_labels = train_labels[idx]

#         # Predizioni della Blackbox
#         blackbox_preds = blackbox_model.predict(real_imgs)

#         # Calcolo della loss e dell'accuratezza del modello pre-addestrato
#         pretrained_loss, pretrained_acc = blackbox_model.evaluate(real_imgs, real_labels, verbose=0)
#         pretrained_losses.append(pretrained_loss)
#         pretrained_accs.append(pretrained_acc)

#         # Addestramento Substitute Detector
#         try:
#             sub_loss, sub_acc = substitute_detector.train_on_batch(real_imgs, blackbox_preds)
#         except Exception as e:
#             log_message(f"Errore durante train_on_batch (Substitute Detector): {e}")
#             sub_loss, sub_acc = 0, 0

#         # Addestramento Generatore
#         noise = tf.convert_to_tensor(np.random.normal(0, 1, (batch_size, latent_dim)), dtype=tf.float32)
#         try:
#             g_loss = train_generator_step(generator, substitute_detector, noise, blackbox_preds, generator_optimizer)
#             g_loss = tf.reduce_mean(g_loss).numpy()
#         except Exception as e:
#             log_message(f"Errore durante train_on_batch (Generatore): {e}")
#             g_loss = 0

#         # Successo del generatore
#         generated_imgs = generator.predict(noise)
#         fake_preds = substitute_detector.predict(generated_imgs)
#         success_rate = 100 * np.mean(np.argmax(fake_preds, axis=1) != np.argmax(blackbox_preds, axis=1))
#         generator_success.append(success_rate)

#         # Log
#         d_losses.append(sub_loss)
#         g_losses.append(g_loss)
#         acc_list.append(sub_acc)
#         log_message(
#             f"Epoch {epoch}: "
#             f"[Sub Loss: {sub_loss:.4f}, Sub Acc: {sub_acc * 100:.2f}%] "
#             f"[Gen Loss: {g_loss:.4f}] [Gen Success: {success_rate:.2f}%] "
#             f"[Blackbox Loss: {pretrained_loss:.4f}, Blackbox Acc: {pretrained_acc * 100:.2f}%]"
#         )

#         if epoch % save_interval == 0:
#             generator.save(f'./generator/generator_epoch_{epoch}.keras')
#             substitute_detector.save(f'./substitute/substitute_detector_epoch_{epoch}.keras')

#     # Grafici
#     plt.figure()
#     plt.plot(d_losses, label="Substitute Detector Loss")
#     plt.plot(g_losses, label="Generator Loss")
#     plt.legend()
#     plt.title("Training Loss")
#     plt.xlabel("Epoch")
#     plt.ylabel("Loss")
#     plt.savefig("./images/loss_plot.png")

#     plt.figure()
#     plt.plot(acc_list, label="Substitute Detector Accuracy")
#     plt.plot(generator_success, label="Generator Success Rate")
#     plt.legend()
#     plt.title("Accuracy and Success Rate")
#     plt.xlabel("Epoch")
#     plt.ylabel("Percentage")
#     plt.savefig("./images/accuracy_and_success_plot.png")

#     # Plot pretrained model performance
#     plt.figure()
#     plt.plot(pretrained_losses, label="Pretrained Model Loss")
#     plt.plot([acc * 100 for acc in pretrained_accs], label="Pretrained Model Accuracy")
#     plt.legend()
#     plt.title("Pretrained Model Performance")
#     plt.xlabel("Epoch")
#     plt.ylabel("Value")
#     plt.savefig("./images/pretrained_model_performance.png")

#     evaluate_and_report(substitute_detector, val_imgs, val_labels, generator=generator, blackbox=blackbox_model)

# # Addestramento
# try:
#     train(epochs=100, batch_size=32, save_interval=5)
# except Exception as e:
#     log_message(f"Errore critico durante l'addestramento: {e}")
