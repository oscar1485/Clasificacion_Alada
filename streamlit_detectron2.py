import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

import numpy as np
import cv2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import streamlit as st

# Configurar Streamlit
st.title("Segmentación de instancias en tiempo real")
st.subheader("Usando Detectron2 y Streamlit")

# Crear la configuración
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Establecer el umbral para este modelo
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.DEVICE = "cpu"  # Usar la CPU en lugar de la GPU

# Crear el predictor
predictor = DefaultPredictor(cfg)

# Inicializar la cámara web
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    st.error("No se pudo acceder a la cámara.")
    st.stop()

# Mapa de nombres de clases del conjunto de datos COCO
coco_classes = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes

# Crear un lugar para visualizar la imagen en Streamlit
image_placeholder = st.empty()

# Botón para detener la ejecución
stop_button = st.button("Detener")

while True:
    # Capturar un fotograma de la cámara web
    ret, frame = cap.read()
    if not ret:
        st.warning("No se pudo capturar la imagen de la cámara.")
        break

    # Realizar la predicción
    outputs = predictor(frame)

    # Obtener las clases detectadas
    classes_detected = outputs["instances"].pred_classes.cpu().numpy()

    # Obtener los nombres de las clases detectadas
    class_names = [coco_classes[i] for i in classes_detected]

    # Visualizar los resultados
    v = Visualizer(frame[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    result = v.get_image()[:, :, ::-1]

    # Convertir la imagen resultante a RGB
    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

    # Mostrar la imagen en Streamlit
    image_placeholder.image(result_rgb, channels="RGB")

    # Verificar si se presionó el botón de detener
    if stop_button:
        break

# Liberar la cámara
cap.release()


