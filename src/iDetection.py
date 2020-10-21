import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import Birch

import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing import image as tf_image
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

###########################################################
################  SUPPORT FUNCTIONS   #####################
###########################################################

def rotate_image(image, angle):
  """
  Función rotar imagen

  Args:
  image
  angle

  Return:
  rotated_image
  """
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result



###########################################################
###################  MAIN FUNCTIONS   #####################
###########################################################
# Detector SIFT
def sift_detector(new_image, image_template):
    """
    Detector SIFT, compara la imagen de referencia y la de estudio.
    Devuelve el número de características SIFT positivas en la comparación.

    Args:
    new_image
    image template

    Return:
    matches
    """
    
    image1 = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
    image2 = image_template

    # Crea el dectector de objetos SIFT
    sift = cv2.SIFT_create()
    # Obtiene los descriptores y puntos clave
    keypoints_1, descriptors_1 = sift.detectAndCompute(image1, None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(image2, None)

    # Define los parámetros del FLANN Matcher
    FLANN_INDEX_KDTREE = 4
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 3)
    search_params = dict(checks = 100)

    # Crea el objeto FLANN Matcher
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Obtiene las comparaciones utilizando K-Nearest Neighbors
    matches = flann.knnMatch(descriptors_1, descriptors_2, k=2)

    # Almacena las comparaciones correctas con el ratio de Lowetest
    good_matches = []
    for m,n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m) 
    return len(good_matches)

def explore_image_SIFT(input_image, image_template, threshold):
    """ 
    Detección y conteo de los objetos de interés

    Args:
        input_image
        image_template
        threshold

    Return
        x_detect
        y_detect    
        score
        counter
    
    """
    height, width = image_template.shape[:2]

    x_detect=[]
    y_detect=[]
    score = []
    #threshold = 15
    counter = 0

    for y in range(0, input_image.shape[0]-height, int(min(image_template.shape[:2])/3)):
        for x in range(0, input_image.shape[1]-width, int(min(image_template.shape[:2])/3)):
            temp_img = input_image[y:y+height, x:x+width]
            matches = sift_detector(temp_img, image_template)
            if matches >= threshold:
                x_detect.append(x)
                y_detect.append(y)
                score.append(matches)
                counter += 1

    return x_detect, y_detect, score, counter

def clustering(x_detect, y_detect, image_template, input_image, k=0.8, label=True):
    """
    Args:
        x_detect
        y_detect
        image_template
        input_image
        k
        label

    Return:
        input_image
        centroids

    """
    height, width = image_template.shape[:2]
    lista = np.array(list(zip(x_detect, y_detect)))
    clustering = Birch(threshold=min(image_template.shape[:2])*k , n_clusters=None).fit(lista)

    centroids = []
    for each in np.unique(clustering.labels_):
        centroids.append(lista[clustering.labels_ == each].sum(axis=0)/len(lista[clustering.labels_ == each]))

    for x, y in centroids:
        x = int(x)
        y = int(y)
        if label:
            cv2.rectangle(input_image, (x, y), (x+width, y+height), (0,255,0), 3)

    return input_image, centroids

def ORB_detector(new_image, image_template):
    """
    Detector ORB, compara la imagen de referencia y la de estudio.
    Devuelve el número de características ORB positivas en la comparación.

    Args:
    new_image
    image template

    Return:
    matches
    """

    image1 = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)

    # Crea el detector ORB con 10000 keypoints y un factor piramidal 1.4
    orb = cv2.ORB_create(10000, 1.4)

    # Detecta los keypoints en la imagen original
    (kp1, des1) = orb.detectAndCompute(image1, None)

    # Detect keypoints of rotated image
    (kp2, des2) = orb.detectAndCompute(image_template, None)

    # Crea el matcher de fuerza bruta
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # Aplica el matcher
    matches = bf.match(des1,des2)

    # Organiza las correspondemcias basada en la distancia de hamming
    matches = sorted(matches, key=lambda val: val.distance)
    return len(matches)


def explore_image_ORB(input_image, image_template, threshold):
    """ 
    Detección y conteo de los objetos de interés

    Args:
        input_image
        image_template
        threshold

    Return
        x_detect
        y_detect    
        score
        counter
    
    """

    height, width = image_template.shape[:2]

    x_detect=[]
    y_detect=[]
    score = []
    # threshold = 4
    counter = 0

    for y in range(0, input_image.shape[0]-height, int(min(image_template.shape[:2])/3)):
        for x in range(0, input_image.shape[1]-width, int(min(image_template.shape[:2])/3)):
            temp_img = input_image[y:y+height, x:x+width]
            matches = ORB_detector(temp_img, image_template)        
            if matches >= threshold:
                x_detect.append(x)
                y_detect.append(y)
                score.append(matches)
                counter += 1                

    return x_detect, y_detect, score, counter

def explore_image_ORB_SIFT(input_image, image_template, threshold_orb, treshhold_sift, k):
    """ 
    Detección y conteo de los objetos de interés

    Args:
        input_image
        image_template
        threshold

    Return
        counter_
        input_image
    
    """

    # ORB
    x_detect, y_detect, score, counter = explore_image_ORB(input_image, image_template, threshold_orb)

    ## Clusterización
    input_image, centroids = clustering(x_detect, y_detect, image_template, input_image, k, False)

    height, width = image_template.shape[:2]

    #SIFT
    x_detect=[]
    y_detect=[]
    score = []
    counter_ = 0
    threshold = 15
    for x, y in centroids:
        x = int(x)
        y = int(y)
        temp_img =input_image[y:y+height, x:x+width]
        matches = sift_detector(temp_img, image_template)
        if matches >= threshold:
            x_detect.append(x)
            y_detect.append(y)
            score.append(matches)
            counter_ += 1
            cv2.rectangle(input_image, (x, y), (x+width, y+height), (0,255,0), 3)


    return counter_, counter, input_image


base_model = InceptionV3(weights='imagenet', include_top=False)

def Inceptionv3_feature(img):
    """
    Obtiene las características de la imagen de acuerdo a Inceptionv3

    Args:
    img

    Return:
    feature_vector
    """
    img = tf.image.resize(img, (224, 224))
    x = tf_image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    return base_model.predict(x)

def Inceptionv3_similarity(image_1, image_2):
    """
    Calcula la similaridad del coseno entre 2 imagenes

    Args:
    image_1
    image_2

    Return:
    cosine_similarity
    """
    I1_vector = Inceptionv3_feature(image_1)
    I2_vector = Inceptionv3_feature(image_2)
    return cosine_similarity(I1_vector.reshape(1,-1), I2_vector.reshape(1,-1))

def explore_image_inceptionv3(input_image, image_template, threshold):
    """ 
    Detección y conteo de los objetos de interés

    Args:
        input_image
        image_template
        threshold

    Return
        x_detect
        y_detect    
        score
        counter
    
    """

    height, width = image_template.shape[:2]

    x_detect=[]
    y_detect=[]
    score = []
    # threshold = 4
    counter = 0

    for y in range(0, input_image.shape[0]-height, int(min(image_template.shape[:2])/2)):
        for x in range(0, input_image.shape[1]-width, int(min(image_template.shape[:2])/2)):
            temp_img = input_image[y:y+height, x:x+width]        
            matches = Inceptionv3_similarity(image_template, temp_img)
            if matches >= threshold:
                x_detect.append(x)
                y_detect.append(y)
                score.append(matches)
                counter += 1           

    return x_detect, y_detect, score, counter