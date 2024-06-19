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





## Resultado

Ver aplicación en: [Streamlit](https://clasificacionalada1.streamlit.app/){:target="_blank"} 

## Agradecimientos

Agradecemos al [Ministerio de Tecnologías de la Información y las Comunicaciones de Colombia](https://mintic.gov.co/ {:target="_blank"}) por financiar la Maestría en Ciencia de Datos.

Asimismo, a la [Universidad Cooperativa de Colombia Campus Ibagué - Espinal](https://www.ucc.edu.co/ target="_blank"){:target="_blank"} por facilitar el apoyo del tiempo dentro del Plan de Trabajo para realizar la Maestría.

Además, a la [Universidad Oberta de Cataluña](https://www.uoc.edu/es){:target="_blank"} por permitir la formación impartida en Ciencia de Datos y la materialización de las competencias aprendidas en este proyecto. A mis tutores Bernat Bas Pujols y Pablo Fernandez Blanco.

Agradecemos el trabajo de: [Krishnaik06](https://github.com/krishnaik06/Deployment-Deep-Learning-Model) además este código queda visible.


