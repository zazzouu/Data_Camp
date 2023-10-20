from PIL import Image
import os
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split  # Ajout de l'importation
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
import numpy as np


# Fonction pour charger les images
def charger_images(chemin_dossier_images, liste_ids):
    images = []
    for img_id in liste_ids:
        img_path = os.path.join(chemin_dossier_images, str(img_id) + '.png')
        img = Image.open(img_path)
        # Redimensionner l'image à la taille souhaitée (par exemple, 224x224)
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0  # Normaliser les pixels
        images.append(img_array)
    return np.array(images)

# Fonction pour prétraiter les images
def preprocess_images(images):
    resized_images = [cv2.resize(img, (224, 224)) for img in images]
    return np.array(resized_images) / 255.0

# Fonction pour l'entraînement du modèle
def entrainement_modele(chemin_train_excel, chemin_eval_excel, chemin_test_excel,
                        chemin_images_train, chemin_images_eval, chemin_images_test):
    # Charger les données d'entraînement, d'évaluation et de test
    labels_train = pd.read_csv(chemin_train_excel)
    labels_eval = pd.read_csv(chemin_eval_excel)
    labels_test = pd.read_csv(chemin_test_excel)

    # Charger les images pour chaque ensemble
    images_train = charger_images(chemin_images_train, labels_train['ID'])
    images_eval = charger_images(chemin_images_eval, labels_eval['ID'])
    images_test = charger_images(chemin_images_test, labels_test['ID'])

    # Prétraiter les images
    X_train = preprocess_images(images_train)
    X_eval = preprocess_images(images_eval)
    X_test = preprocess_images(images_test)

    # Les labels
    y_train = labels_train['Disease_Risk']
    y_eval = labels_eval['Disease_Risk']
    y_test = labels_test['Disease_Risk']

    # Créer le modèle CNN
    model = Sequential([
        Conv2D(64, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    # Compiler le modèle
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Entraîner le modèle
    model.fit(X_train, y_train, validation_data=(X_eval, y_eval), epochs=3, batch_size=32)

    # Évaluer le modèle
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print("Précision sur l'ensemble de test :", test_acc)

    # Sauvegarder le modèle pour une utilisation future
    model.save('modele_cancer_des_yeux.h5')





chemin_vers_excel_train= 'eyes-dataset/Training_Set/Training_Set/RFMiD_Training_Labels.csv'
chemin_vers_excel_evaluation='eyes-dataset/Evaluation_Set/Evaluation_Set/RFMiD_Validation_Labels.csv'
chemin_vers_excel_test='eyes-dataset/Test_Set/Test_Set/RFMiD_Testing_Labels.csv'
chemin_vers_dossier_images_train='eyes-dataset/Training_Set/Training_Set/Training/'
chemin_vers_dossier_images_evaluation='eyes-dataset/Evaluation_Set/Evaluation_Set/Validation/'
chemin_vers_dossier_images_test='eyes-dataset/Test_Set/Test_Set/Test/'



entrainement_modele(chemin_vers_excel_train, chemin_vers_excel_evaluation, chemin_vers_excel_test,
                    chemin_vers_dossier_images_train, chemin_vers_dossier_images_evaluation, chemin_vers_dossier_images_test)


