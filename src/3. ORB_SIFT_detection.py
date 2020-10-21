import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import Birch

from iDetection import *


## Imagen Original
PATH = "../data/JPEG/" 
image = cv2.imread(PATH+'IMG_2465.jpg')
print('Tamaño original : ', image.shape)
 
scale_percent = 50 
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)
dim = (width, height)

resized_image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
 
print('Tamaño reducido : ',resized_image.shape)
 
plt.axis("off")
plt.imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
plt.show()

# Selección del objeto de interés
cropped = rotate_image(resized_image,0)[ int(770*scale_percent/100):int(950*scale_percent/100),
                                           int(630*scale_percent/100):int(760*scale_percent/100)]

plt.axis("off")
plt.imshow(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
plt.show()

# Detección y conteo de los objetos de interés
# ORB_SIFT
counter_, counter, input_image = explore_image_ORB_SIFT(resized_image, cropped, 2, 15, 0.4)



## Resultados
print(f'El número de objetos detectados antes de la clusterización es : ', str(counter))
print(f"El número de objetos detectados fue: ",counter_)

font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(input_image,'Objetos detectados:'+str(counter_),(10,100), font, 2,(0,0,255),6)

plt.imshow(input_image)
plt.show()

cv2.imwrite('ORB_SIFT_DETECTION.jpg', input_image);