# Prueba Técnica - Deasarrollador Machine Learning (Visón por computador)

El objetivo de éste proyecto es mostrar diferentes técnicas y algoritmos para trabajar con diferentes imágenes y detectar objetos en ella.

## Tecnlogogía escogida
* **Lenguaje de programación:** El leguaaje de programación seleccionado es **Python** en su version 3.6. La razón de ésta selección es la versatilidad del lenguaje, su simplicidad  y compatibilidad con diversas librerías relacionadas al área del machine learning, deep learning y visión de máquina.
* **Principales Librerías/Frameworks utilizados:**
    * **OpenCV:** Utilizada ampliamente en el área de visión de máquina y procesamiento de imágenes. Posee diversas funcionalidades que facilitan el trabajo con imágenes, principalmente en métodos tradicionales (no deep learning).
    * **Numpy:** Libería matemática, relacionada con operaciones de álgebra lineal. Permite el diferentes operaciones con matrices y vectores.
    * **Matplotlib:** Librería de visualización de datos e imágenes.
    * **Scikitlearn:**  Utilizada ampliamente en el área del machine learning, ya que permite un uso sencillo de múltiples algoritmos.

## Algoritmos Utilizados 
Se han propuesto diferentes algoritmos que indican diferentes formas de interacción con las imágenes detalladas

### **1. Scale-invariant feature transform (SIFT)**
En este algoritmo se utiliza la técnica SIFT para extraer las características más relevantes de uma imagen patrón a detectar.

# **Add SIFT image**

El detector SIFT sigue la estructura:

    1. Obtener las imagenes de referencia y comparación (las imagenes )
    2. La imagen de comparación es convertida a escala de grises
    3. Se crea el detector de objetos SIFT
    4. Se obtienes los descriptores y puntos clave en cada una de las imagenes
    5. Se define el Fast Library for Aproximate Newarest Neighbors Matcher, el cual contiene algoritmos optimizados para una búsqueda rápida de nearest neighbors (vecinos cercanos) en bases de datos con alta dimensionalidad. Se especifica un árbol de decisión como clasificador y su número de checks (100).
    6. Se utiliza el classificador propuesto para comparar los descriptores de las figuras en estudio.
    7. Se comparan los resultados obtenidos utilizando la métrica de Lowe (Lowe rate), que compara la distancia entre los descriptores. Se almacenan los mejores resultados.
    
En este programa se propuso:

    1. Reducción del tamaño de la imagen en un 50% (reducción de costo computacional).
    2. Transformación a escala de grises.
    3. Recortar el área de interés (donde se encuentre el objeto que quiero detectar).
    4. Se recorre la imagen de estudio en pequeñas secciones. 
        4.1 En cada subsección se aplica el detector SIFT para comparar cada subsección en estudio y la imagen de referencia.
        4.2 Las subsecciones que superen el treshold (umbral) de similitud basada en el SIFT son almacenadas.
    5. Clusterización de detecciones sobrepuestas en un mismo objeto. Es utilzado el algoritmo BIRCH (Balanced Iterative Reducing and Clustering using Hierarchies) debido as su efficiencia de uso de memoria y a su capacidad de selección de número de clusters basado en distancia (relacionada con el objeto de interés a ser detectado).
    6. Con el número de clusters encontrado, cada cluster es promediado para encontrar un valor medio por cluster (una forma de centroide del cluster) e indicar así la ubicación de cada uno de los objetos detectados.
    7. Los clusters son mostrados en la imagen de estudio, identificando el objeto de interés.


### **2. Oriented FAST and Rotated BRIEF (ORB)**

En este algoritmo se utiliza la técnica ORB para extraer las características más relevantes de una imagen patrón a detectar. 
Su ventaja con respecto a SIFT es su mayor eficiencia computacional. 

# **Add ORB image**

El detector ORB sigue la estructura:

    1. Obtener las imagenes de referencia y comparación (las imagenes )
    2. La imagen de comparación es convertida a escala de grises
    3. Se crea el detector de objetos ORB, definiendo 10000 características a ser retenidas y su pirámide de escala (1.4) [representación multiescala de una sóla imagen].
    4. Se obtienes los descriptores y puntos clave en cada una de las imagenes
    5. Se define el matcher de Fuerza Bruta o BFMatcher para comparar el descriptor de una característica de un conjunto y se compara con las demas características utilizando en este caso la distancia de Hamming. Los resultados son organizados de acuerdo a su desempeño (menor distancia es mejor) y se devuelve el número de características similares encontradas.
    
En este programa se propuso:

    1. Reducción del tamaño de la imagen en un 50% (reducción de costo computacional).
    2. Transformación a escala de grises.
    3. Recortar el área de interés (donde se encuentre el objeto que quiero detectar).
    4. Se recorre la imagen de estudio en pequeñas secciones. 
        4.1 En cada subsección se aplica el detector ORB para comparar cada subsección en estudio y la imagen de referencia.
        4.2 Las subsecciones que superen el treshold (umbral) de similitud basada en el ORB son almacenadas.
    5. Clusterización de detecciones sobrepuestas en un mismo objeto. Es utilzado el algoritmo BIRCH (Balanced Iterative Reducing and Clustering using Hierarchies) debido as su efficiencia de uso de memoria y a su capacidad de selección de número de clusters basado en distancia (relacionada con el objeto de interés a ser detectado).
    6. Con el número de clusters encontrado, cada cluster es promediado para encontrar un valor medio por cluster (una forma de centroide del cluster) e indicar así la ubicación de cada uno de los objetos detectados.
    7. Los clusters son mostrados en la imagen de estudio, identificando el objeto de interés.


### **3. ORB y SIFT (Técnica Híbrida)**

En este algoritmo se propone utilizar las técnicas ORB y SIFT para extraer las características más relevantes de una imagen patrón a detectar.
Se toman la ventaja computacional de ORB para detectar posibles subsecciones candidatas, y la buena capacidad de comparación de SIFT para avalar si esas subsecciones pertenecen o no al patrón detectado.

El detector ORB y SIFT siguen las estructuras mostradas previamente.
    
En este programa se propuso:

    1. Reducción del tamaño de la imagen en un 50% (reducción de costo computacional).
    2. Transformación a escala de grises.
    3. Recortar el área de interés (donde se encuentre el objeto que quiero detectar).
    4. Se recorre la imagen de estudio en pequeñas secciones. 
        4.1 En cada subsección se aplica el detector ORB para comparar cada subsección en estudio y la imagen de referencia.
        4.2 Las subsecciones que superen el treshold (umbral) de similitud basada en el ORB son almacenadas.
    5. Clusterización de detecciones sobrepuestas en un mismo objeto. Es utilzado el algoritmo BIRCH (Balanced Iterative Reducing and Clustering using Hierarchies) debido as su efficiencia de uso de memoria y a su capacidad de selección de número de clusters basado en distancia (relacionada con el objeto de interés a ser detectado).
    6. Con el número de clusters encontrado, cada cluster es promediado para encontrar un valor medio por cluster (una forma de centroide del cluster) e indicar así la ubicación de cada uno de los objetos detectados.  
    7. Cada uno de Los clusters encontrados es analizado utilizando SIFT para detectar si efectivamente los clusters indicador por ORB pertenecen al conjunto de imágenes con alta similitud a la imagen patrón.
    8. Los clusters aceptados por el algoritmo SIFT son mostrados en la imagen base.




## Diagramas de flujo





# Problemas Encontrados y como fueron solucionados
1. Las imagenes .HEIC (propios de dispositivos Apple - iPhone) presentaron una limitación para su uso en Windows y Linux.

    **Solucion:** Se convirtieron las imágenes a un formato compatible (PNG). Se desarrolló un script que permitió esa conversión utilizando Google Colab (script ubicado en /src/HEIC2JPG_colab.py ).
2. Sobreposición de múltiples áreas de detección para la identificación de un mismo objeto.

    **Solucion:** Se utilizó un algoritmo de *clusterización* basado en la distancia entre objetos para agrupar las detecciones contiguas.

3. Selección de la imágen patrón (ejemplo) y el *treshold* (umbral) que permiten la detección de items similares. Diferentes patrones y *tresholds* pueden cambiar completamente el resultado del algoritmo.

    **Solucion:** Se buscó  de forma heurística un patrón y un *treshold* que mostrase buenos resultados.

4. Selección de la imágen patrón (ejemplo) para detectar los demás items similares, diferentes patrones en la misma imagen cambian el resultado del algoritmo.

    **Solucion:** Se buscó  de forma heurística un patrón que mostrase buenos resultados.



# Posibles mejoras al algoritmo propuesto
* Mejoras del algoritmo de clusterización para obtener mejores resultados en la detección.
* Ajuste de hiperparámetros (constantes y métodos) de los algoritmos propuestos, ya que estas mejoras finas pueden mejorar considerablemente el desempeño del detector.
* Uso de técnicas híbridas (uso de *ensemble methods*) para mejorar el desempeño de los algoritmos.
* Detección de espacios vacíos en los estantes.
* Uso de ténicas más complejas y recientes.
* Recolección de mayores cantidad de imágenes para proponer un abordaje más moderno del problema (deep learning).


# Soporte gráfico de los resultados