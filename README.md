# CLASIFICACIÓN ALADA

Implementaremos una red neuronal utilizando Keras y TensorFlow, y la ejecutaremos en un servicio web con Flask y Streamlit.

## 1. Preparación del entorno
    $ conda create -n TFMaves anaconda python=3.9.19
    $ conda activate FMaves
    $ conda install ipykernel
    $ python -m ipykernel install --user --name FMavesp --display-name FMaves"
    $ pip install -r requirements.txt


Asegúrate de tener un archivo `requirements.txt` con las siguientes dependencias:

    $ Flask==3.0.3
    $ tensorflow
    $ keras==3.1.1
    $ numpy==1.26.2
    $ pandas==2.2.1
    $ opencv-python==4.9.0.80
    $ gunicorn==20.1.0
    $ load_dotenv
    $ streamlit
    $ scikit-learn
    $ scikit-image
    $ opencv-python-headless
    $ pillow==10.3.0
    $ matplotlib==3.8.3
    $ scipy==1.12.0
    $ requests==2.31.0
    $ h5py==3.11.0

## 2. Entrenar la red neuronal

1.  Descargar el repositorio.

2.  Abrir una terminal en la carpeta del proyecto y ejecutar Jupyter Notebook:
Copiar código
```         
`$ jupyter notebook`
```

3.  Ejecutar el notebook `BirdClass_ajuste_fino.ipynb` para entrenar la red neuronal Y guardar el mejor modelo.

## 3. Probar la red neuronal

Para probar el modelo entrenado, ejecutar:
Copiar código
```         
`$ python TestModel.py`
```

## 4. Probar el API de Flask

Para iniciar el servicio web con Flask, ejecutar:
Copiar código

```         
`$ python app.py`
```


## 5. Probar la interfaz de usuario con Streamlit

Para iniciar la interfaz de usuario con Streamlit, ejecutar:
Copiar código
```         
`$ streamlit run streamlit_app.py`
```
## 6. Resultado

Ver aplicación en: [Streamlit](https://clasificacionalada1.streamlit.app/)

## 7. Estructura de Directorios y Archivos del Proyecto

### 7.1 Directorios

- **datasetpreprocesado/**: Contiene las imágenes por clase preprocesadas utilizadas para entrenar y evaluar los modelos de clasificación de aves.
  
- **models/**: Almacena los modelos preentrenados y otros relacionados con el proyecto, como el archivo optimizado.keras.

- **output/**: Contiene imágenes que se utilizan para pruebas del modelo.

- **static/**: Almacena archivos estáticos como CSS y JavaScript utilizados por la aplicación Flask.

- **templates/**: Contiene las plantillas HTML usadas por Flask para renderizar las páginas web.

- **uploads/**: Directorio donde se guardan temporalmente las imágenes subidas por los usuarios para predicciones.

### 7.2 Archivos

- **appFlask.py**: Script de Python que contiene la aplicación web basada en Flask para la clasificación de aves.

- **appStreamlit.py**: Script de Python que contiene la aplicación web basada en Streamlit para la clasificación de aves.

- **BirdClass_ajuste_fino.html**: Exportación en formato HTML de un notebook de Jupyter (.ipynb) que probablemente contiene el proceso de ajuste fino del modelo de clasificación de aves.

- **BirdClass_ajuste_fino.ipynb**: Notebook de Jupyter que contiene el código y documentación sobre el ajuste fino del modelo de clasificación de aves.

- **Dockerfile**: Archivo de configuración para Docker que define cómo construir una imagen de Docker para el proyecto, especificando el entorno cuando se trabaja con Flask y se quiere publicar en línea.

- **ImagenPrueba.jpg** y **ImagenPrueba_sin_fondo.jpg**: Imágenes de prueba que pueden ser utilizadas para validar y demostrar las predicciones del modelo.

- **LIME.html**: Exportación en formato HTML de un notebook de Jupyter (.ipynb) relacionado con LIME (Local Interpretable Model-agnostic Explanations), una técnica para explicar las predicciones del modelo.

- **LIME.ipynb**: Notebook de Jupyter que contiene el código y documentación sobre la aplicación de LIME para explicar las predicciones del modelo de clasificación de aves.

- **README.md**: Archivo de Markdown que proporciona una descripción general del proyecto, incluyendo instrucciones de uso, instalación y propósito.

- **requirements.txt**: Archivo que lista todas las dependencias y bibliotecas de Python necesarias para el proyecto, que se pueden instalar usando pip.

- **TestModel.py**: Script de Python que probablemente contiene pruebas para validar el modelo.


## 8. Agradecimientos

Agradecemos al <a href="https://mintic.gov.co/" target="_blank">Ministerio de Tecnologías de la Información y las Comunicaciones de Colombia</a> por financiar la Maestría en Ciencia de Datos.

Asimismo, a la <a href="https://www.ucc.edu.co/" target="_blank">Universidad Cooperativa de Colombia Campus Ibagué - Espinal</a> por facilitar el apoyo del tiempo dentro del Plan de Trabajo para realizar la Maestría.

Además, a la <a href="https://www.uoc.edu/es" target="_blank">Universidad Oberta de Cataluña</a> por permitir la formación impartida en Ciencia de Datos y la materialización de las competencias aprendidas en este proyecto. A mis tutores Bernat Bas Pujols y Pablo Fernandez Blanco.



