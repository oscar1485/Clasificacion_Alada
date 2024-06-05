# CLASIFICACIÓN ALADA

Implementaremos una red neuronal utilizando Keras y TensorFlow, y la ejecutaremos en un servicio web con Flask y Streamlit.

## 1. Preparación del entorno
    $ conda create -n TFMaves anaconda python=3.9.19
    $ conda activate FMaves
    $ conda install ipykernel
    $ python -m ipykernel install --user --name FMavesp --display-name FMaves"
    $ pip install -r requirements.txt


Asegúrate de tener un archivo `requirements.txt` con las siguientes dependencias:

$ tensorflow==2.16.1
$ tensorflow-intel==2.16.1
$ keras==3.1.1
$ h5py==3.11.0
$ flask==3.0.3
$ gunicorn==20.1.0
$ streamlit==1.35.0
$ opencv-python==4.9.0.80
$ opencv-python-headless==4.10.0.82
$ scikit-image==0.22.0
$ pillow==10.3.0
$ numpy==1.26.2
$ pandas==2.2.1
$ matplotlib==3.8.3
$ scipy==1.12.0
$ requests==2.31.0

## 2. Entrenar la red neuronal

1.  Descargar el repositorio.

2.  Abrir una terminal en la carpeta del proyecto y ejecutar Jupyter Notebook:

```         
bash
```

Copiar código

`$ jupyter notebook`

3.  Ejecutar el notebook `BirdClass.ipynb` para entrenar la red neuronal.

## 3. Probar la red neuronal

Para probar el modelo entrenado, ejecutar:

```         
bash
```

Copiar código

`$ python TestModel.py`

## 4. Probar el API de Flask

Para iniciar el servicio web con Flask, ejecutar:

```         
bash
```

Copiar código

`$ python app.py`

## 5. Probar la interfaz de usuario con Streamlit

Para iniciar la interfaz de usuario con Streamlit, ejecutar:

```         
bash
```

Copiar código

`$ streamlit run streamlit_app.py`

## Resultado

## Agradecimientos

Agradecemos el trabajo de: [Krishnaik06](https://github.com/krishnaik06/Deployment-Deep-Learning-Model)

# **Canal de Youtube**

[Haz clic aquí para ver mi canal de YouTube](https://www.youtube.com/channel/UCr_dJOULDvSXMHA1PSHy2rg)

Este ajuste proporciona instrucciones actualizadas y claras para preparar el entorno, entrenar y probar la red neuronal, así como para iniciar el servicio web con Flask y la interfaz de usuario con Streamlit.
