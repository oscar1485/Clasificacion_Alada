import tqdm
tqdm.tqdm.monitor_interval = 0  # Desactiva la barra de progreso

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

import os
from keras.models import load_model
from keras.applications.imagenet_utils import preprocess_input
import numpy as np
import cv2
import torch
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2 import model_zoo
import streamlit as st
from PIL import Image
from skimage.transform import resize
import pandas as pd

# Definir clases de aves
bird_classes = [14, 15, 16]  # Clases de aves en COCO dataset

# Función para detectar aves en la imagen usando Detectron2
def detect_birds(img):
    # Convertir la imagen a RGB si tiene cuatro canales (RGBA)
    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

    # Crear la configuración y cargar el modelo preentrenado
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.97  # Umbral de confianza para la detección
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.DEVICE = "cpu"  # Usar CPU en lugar de GPU
    predictor = DefaultPredictor(cfg)

    # Realizar la predicción
    outputs = predictor(img)
    classes_detected = outputs["instances"].pred_classes.cpu().numpy()

    # Verificar si se detecta algún pájaro en la imagen
    bird_detected = any(cls in bird_classes for cls in classes_detected)

    return bird_detected

# Resto del código Streamlit
MODEL_PATH = 'models/optimizado.keras'
EXCEL_PATH = 'aves.xlsx'

width_shape = 224
height_shape = 224

names = ['CATHARTES AURA', 'COEREBA FLAVEOLA', 'COLUMBA LIVIA', 'CORAGYPS ATRATUS', 'CROTOPHAGA SULCIROSTRIS', 'CYANOCORAX YNCAS',
         'EGRETTA THULA', 'FALCO PEREGRINUS', 'FALCO SPARVERIUS', 'HIRUNDO RUSTICA', 'PANDION HALIAETUS', 'PILHERODIUS PILEATUS',
         'PITANGUS SULPHURATUS', 'PYRRHOMYIAS CINNAMOMEUS', 'RYNCHOPS NIGER', 'SETOPHAGA FUSCA', 'SYNALLAXIS AZARAE', 'TYRANNUS MELANCHOLICUS']

def model_prediction(img, model):
    img_resize = resize(img, (width_shape, height_shape))
    x = preprocess_input(img_resize * 255)
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x)
    return preds

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
    try:
        model = load_model(MODEL_PATH)
        st.success("Modelo cargado correctamente")
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        return

    st.image("banner2.jpg", use_column_width=True)
    st.title("Clasificación Alada2")
    st.header("Sistema Multiclase para la Identificación Aviar en Ibagué")

    menu = ["Información del Proyecto", "Realizar Predicciones", "Listar Aves Entrenadas", "Agradecimientos"]
    choice = st.sidebar.selectbox("Selecciona una opción", menu)

    if choice == "Realizar Predicciones":
        st.subheader("Realizar Predicciones")
        img_file_buffer = st.file_uploader("Carga una imagen", type=["png", "jpg", "jpeg"])

        if img_file_buffer is not None:
            image = np.array(Image.open(img_file_buffer))
            st.image(image, caption="Imagen", use_column_width=True)

            # Validar si la imagen contiene aves usando Detectron2
            bird_detected = detect_birds(image)

            if bird_detected:
                if st.button("Identificar Ave"):
                    preds = model_prediction(image, model)
                    bird_name = names[np.argmax(preds)]
                    confidence = np.max(preds)
                    st.success(f'El ave es: {bird_name} con una precisión del {confidence:.2%}')

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
                    st.warning("La imagen corresponde a un Ave, ya puedes dar clic en el botón para realizar la predicción.")
            else:
                st.warning("La imagen no contiene pájaros. No se puede realizar la predicción. Por favor, carga una imagen que contenga un ave.")

        else:
            st.warning("Por favor, carga una imagen primero.")

    elif choice == "Listar Aves Entrenadas":
        st.subheader("Listado de Aves Entrenadas")
        birds_info = [
            {"name": "CATHARTES+AURA", "image": "static/imagen/CATHARTES AURA_2.jpg"},
            {"name": "COEREBA+FLAVEOLA", "image": "static/imagen/COEREBA FLAVEOLA_1.jpg"},
            {"name": "COLUMBA+LIVIA", "image": "static/imagen/COLUMBA LIVIA_7.jpg"},
            {"name": "CORAGYPS+ATRATUS", "image": "static/imagen/CORAGYPS ATRATUS_10.jpg"},
            {"name": "CROTOPHAGA+SULCIROSTRIS", "image": "static/imagen/CROTOPHAGA SULCIROSTRIS_7.jpg"},
            {"name": "CYANOCORAX+YNCAS", "image": "static/imagen/CYANOCORAX YNCAS_8.jpg"},
            {"name": "EGRETTA+THULA", "image": "static/imagen/EGRETTA THULA_1.jpg"},
            {"name": "FALCO+PEREGRINUS", "image": "static/imagen/FALCO PEREGRINUS_6.jpg"},
            {"name": "FALCO+SPARVERIUS", "image": "static/imagen/FALCO SPARVERIUS_1.jpg"},
            {"name": "HIRUNDO+RUSTICA", "image": "static/imagen/HIRUNDO RUSTICA_10.jpg"},
            {"name": "PANDION+HALIAETUS", "image": "static/imagen/PANDION HALIAETUS_5.jpg"},
            {"name": "PILHERODIUS+PILEATUS", "image": "static/imagen/PILHERODIUS PILEATUS_5.jpg"},
            {"name": "PITANGUS+SULPHURATUS", "image": "static/imagen/PITANGUS SULPHURATUS_6.jpg"},
            {"name": "PYRRHOMYIAS+CINNAMOMEUS", "image": "static/imagen/PYRRHOMYIAS CINNAMOMEUS_3.jpg"},
            {"name": "RYNCHOPS+NIGER", "image": "static/imagen/RYNCHOPS NIGER_5.jpg"},
            {"name": "SETOPHAGA+FUSCA", "image": "static/imagen/SETOPHAGA FUSCA_10.jpg"},
            {"name": "SYNALLAXIS+AZARAE", "image": "static/imagen/SYNALLAXIS AZARAE_7.jpg"},
            {"name": "TYRANNUS+MELANCHOLICUS", "image": "static/imagen/TYRANNUS MELANCHOLICUS_7.jpg"}
        ]
        for bird in birds_info:
            bird_name = bird["name"].replace('+', ' ')
            bird_image = bird["image"]
            st.image(bird_image, caption=bird_name)
    
    elif choice == "Información del Proyecto":
        st.subheader("Información del Proyecto")
        st.write("Descripción del proyecto...")
    
    elif choice == "Agradecimientos":
        st.subheader("Agradecimientos")
        st.write("Agradecimientos...")

if __name__ == "__main__":
    main()
