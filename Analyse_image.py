
from keras.preprocessing import image
import numpy as np
import cv2
from keras.models import load_model



# Importez le modèle ici
modele = load_model('modele_cancer_des_yeux.h5')


def traiter_image(filename):
    if filename.endswith('.png'):
        try:
            img = image.load_img(filename, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = img_array / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            predictions = modele.predict(img_array)
            img_cv2 = cv2.imread(filename)
            probability_cancer_detected = predictions[0][0]
            seuil = 0.5
            if probability_cancer_detected > seuil:
                cv2.putText(img_cv2, "Cancer detecte", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imwrite(filename, img_cv2)
                return "Cancer detecté"
            else:
                cv2.putText(img_cv2, "Pas de cancer detecte", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imwrite(filename, img_cv2)
                return "Pas de cancer detecté"
            
        except Exception as e:
            return f'Erreur lors du traitement de l\'image : {str(e)}'
    else:
        return 'Seules les images PNG sont autorisées.'
