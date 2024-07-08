from keras.models import load_model
from keras.applications.imagenet_utils import preprocess_input
import numpy as np
import streamlit as st
from PIL import Image
from skimage.transform import resize
import pandas as pd
import openpyxl
import os

# Path del modelo preentrenado
MODEL_PATH = 'models/optimizado.keras'

# Path del archivo Excel
EXCEL_PATH = 'aves.xlsx'

# Dimensiones de las imágenes de entrada    
width_shape = 224
height_shape = 224

# Clases
names = ['CATHARTES AURA', 'COEREBA FLAVEOLA', 'COLUMBA LIVIA', 'CORAGYPS ATRATUS','CROTOPHAGA SULCIROSTRIS', 'CYANOCORAX YNCAS',
         'EGRETTA THULA', 'FALCO PEREGRINUS','FALCO SPARVERIUS', 'HIRUNDO RUSTICA', 'PANDION HALIAETUS', 'PILHERODIUS PILEATUS',
         'PITANGUS SULPHURATUS','PYRRHOMYIAS CINNAMOMEUS', 'RYNCHOPS NIGER', 'SETOPHAGA FUSCA','SYNALLAXIS AZARAE', 'TYRANNUS MELANCHOLICUS']

# Se recibe la imagen y el modelo, devuelve la predicción
def model_prediction(img, model):
    img_resize = resize(img, (width_shape, height_shape))
    x = preprocess_input(img_resize * 255)
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x)
    return preds

# Función para obtener información adicional del ave desde el archivo Excel
def get_bird_info(bird_name, excel_path):
    df = pd.read_excel(excel_path)
    bird_info = df[df['Nombre_Cientifico'] == bird_name]
    if not bird_info.empty:
        bird_info = bird_info.iloc[0]  # Selecciona la primera fila (debería ser única)
        return bird_info
    else:
        return None

def load_bird_images(bird_name):
    bird_dir = os.path.join('datasetpreprocesado/test', bird_name.replace(' ', ' '))
    
    # Mostrar la ruta que está siendo intentada
    st.write(f"Intentando acceder a la ruta: {bird_dir}")
    
    bird_name_buscar = bird_name.replace(' ', '+')
    st.markdown(f"[Ver más Información](https://www.google.com/search?q={bird_name_buscar})")
    
    if os.path.exists(bird_dir):
        images = []
        for img_file in os.listdir(bird_dir):
            if img_file.endswith(('png', 'jpg', 'jpeg')):
                img_path = os.path.join(bird_dir, img_file)
                images.append(img_path)
        return images
    else:
        return []

def main():
    st.title("Clasificación Alada")
    st.header("Sistema Multiclase para la Identificación Aviar en Ibagué")
    st.markdown("""
    ### Información del Proyecto
    El proyecto "Clasificación Alada" es un sistema multiclase diseñado para la identificación de aves en la región de Ibagué, 
    con un enfoque centrado en técnicas de Deep Learning. El objetivo principal es proporcionar una herramienta precisa y eficiente 
    para la clasificación de aves a partir de imágenes. Este trabajo fue apoyado en parte por Ministerio de Tecnologías de la Información y las Comunicaciones de Colombia y la Universidad Cooperativa de Colombia, Campus Ibagué - Espinal.

    
    **Oscar Augusto Diaz Triana**  
    Universidad Oberta de Cataluña  
    Master en Ciencia de Datos  
    Deep Learning  
    **Tutores de TF:** Bernat Bas Pujols, Pablo Fernandez Blanco  
    **Profesor responsable de la asignatura:** Albert Solé  
    **2024**
    """)

    # Menú
    menu = ["Información del Proyecto", "Realizar Predicciones", "Listar Aves Entrenadas", "Agradecimientos"]
    choice = st.sidebar.selectbox("Selecciona una opción", menu)

    # Se intenta cargar el modelo
    try:
        model = load_model(MODEL_PATH)
        st.success("Modelo cargado correctamente")
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        return

    if choice == "Realizar Predicciones":
        st.subheader("Realizar Predicciones")
        img_file_buffer = st.file_uploader("Carga una imagen", type=["png", "jpg", "jpeg"])
        
        if img_file_buffer is not None:
            image = np.array(Image.open(img_file_buffer))    
            st.image(image, caption="Imagen", use_column_width=True)
        
        if st.button("Identificar Ave"):
            if img_file_buffer is not None:
                predictS = model_prediction(image, model)
                bird_name = names[np.argmax(predictS)]
                st.success(f'El ave es: {bird_name}')

                # Buscar información del ave en el archivo Excel
                bird_info = get_bird_info(bird_name, EXCEL_PATH)
                if bird_info is not None:
                    st.write("**Nombre Científico:**", bird_info['Nombre_Cientifico'])
                    st.write("**Nombre Común:**", bird_info['Nombre_Comun'])
                    st.write("**Descripción General:**", bird_info['Descripcion_General'])
                    st.write("**Distribución en el Tolima:**", bird_info['Distribucion_tolima'])
                    st.write("**Distribución en Colombia:**", bird_info['Distribucion_Colombia'])
                    st.write("**Estado de Conservación:**", bird_info['Estado_Conservacion'])
                else:
                    st.warning("No se encontró información adicional sobre esta ave.")
                
                # Cargar y mostrar imágenes del ave desde el directorio correspondiente
                bird_images = load_bird_images(bird_name)
                if bird_images:
                    st.subheader("Galería de Imágenes del Ave")
                    cols = st.columns(3)
                    for i, img_path in enumerate(bird_images):
                        img = Image.open(img_path)
                        cols[i % 3].image(img, use_column_width=True)
                else:
                    st.warning("No se encontraron imágenes adicionales del ave en la galería.")
            else:
                st.warning("Por favor, carga una imagen primero.")

    elif choice == "Listar Aves Entrenadas":
        st.subheader("Listar Aves Entrenadas")
        birds_info = [
            {"name": "CATHARTES+AURA", "image": "static/imagen/CATHARTES AURA_7.jpg"},
            {"name": "COEREBA+FLAVEOLA", "image": "static/imagen/COEREBA FLAVEOLA.jpg"},
            {"name": "COLUMBA+LIVIA", "image": "static/imagen/COLUMBA LIVIA_9.jpg"},
            {"name": "CORAGYPS+ATRATUS", "image": "static/imagen/CORAGYPS ATRATUS_19.jpg"},
            {"name": "CROTOPHAGA+SULCIROSTRIS", "image": "static/imagen/CROTOPHAGA SULCIROSTRIS_3.jpg"},
            {"name": "CYANOCORAX+YNCAS", "image": "static/imagen/CYANOCORAX YNCAS_3.jpg"},
            {"name": "EGRETTA+THULA", "image": "static/imagen/EGRETTA THULA_1.jpg"},
            {"name": "FALCO+PEREGRINUS", "image": "static/imagen/FALCO PEREGRINUS_9.jpg"},
            {"name": "FALCO+SPARVERIUS", "image": "static/imagen/FALCO SPARVERIUS_17.jpeg"},
            {"name": "HIRUNDO+RUSTICA", "image": "static/imagen/HIRUNDO RUSTICA_10.jpg"},
            {"name": "PANDION+HALIAETUS", "image": "static/imagen/PANDION HALIAETUS_5.jpg"},
            {"name": "PILHERODIUS+PILEATUS", "image": "static/imagen/PILHERODIUS PILEATUS_14.jpeg"},
            {"name": "PITANGUS+SULPHURATUS", "image": "static/imagen/PITANGUS SULPHURATUS_12.jpg"},
            {"name": "PYRRHOMYIAS+CINNAMOMEUS", "image": "static/imagen/PYRRHOMYIAS CINNAMOMEUS_14.jpg"},
            {"name": "RYNCHOPS+NIGER", "image": "static/imagen/RYNCHOPS NIGER_9.jpg"},
            {"name": "SETOPHAGA+FUSCA", "image": "static/imagen/SETOPHAGA FUSCA_5.jpg"},
            {"name": "SYNALLAXIS+AZARAE", "image": "static/imagen/SYNALLAXIS AZARAE_17.jpeg"},
            {"name": "TYRANNUS+MELANCHOLICUS", "image": "static/imagen/TYRANNUS MELANCHOLICUS_12.jpg"},
        ]

        num_columns = 3
        num_rows = int(np.ceil(len(birds_info) / num_columns))

        for i in range(num_rows):
            bird_row = birds_info[i * num_columns: (i + 1) * num_columns]

            # Crear una fila de la tabla
            col1, col2, col3 = st.columns(3)

            for j, bird in enumerate(bird_row):
                # Agregar imagen, nombre y enlace a Google a cada columna
                if j == 0:
                    with col1:
                        st.image(bird["image"], caption=bird["name"], width=100)
                        st.write(bird["name"])
                        st.markdown(f"[Buscar en Google](https://www.google.com/search?q={bird['name']})")
                elif j == 1:
                    with col2:
                        st.image(bird["image"], caption=bird["name"], width=100)
                        st.write(bird["name"])
                        st.markdown(f"[Buscar en Google](https://www.google.com/search?q={bird['name']})")
                elif j == 2:
                    with col3:
                        st.image(bird["image"], caption=bird["name"], width=100)
                        st.write(bird["name"])
                        st.markdown(f"[Buscar en Google](https://www.google.com/search?q={bird['name']})")

    elif choice == "Agradecimientos":
        st.subheader("Agradecimientos")
        st.markdown("""
        Agradezco al Ministerio de Tecnologías de la Información y las Comunicaciones de Colombia por financiar la Maestría en Ciencia de Datos. 
        Asimismo, a la Universidad Cooperativa de Colombia Campus Ibagué - Espinal por facilitar el apoyo del tiempo dentro del Plan de Trabajo para realizar la Maestría. 
        Además, a la Universidad Oberta de Cataluña por permitir la formación impartida y la materialización de las competencias aprendidas en este proyecto, a mis tutores Bernat Bas Pujols y Pablo Fernandez Blanco.
        """)

if __name__ == '__main__':
    main()
