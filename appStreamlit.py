from keras.models import load_model
from keras.applications.imagenet_utils import preprocess_input
import numpy as np
import streamlit as st
from PIL import Image
from skimage.transform import resize
import cv2

# Path del modelo preentrenado
MODEL_PATH = 'models/optimizado.keras'

# Path del archivo Excel
EXCEL_PATH = 'aves.xlsx'

# Dimensiones de las imagenes de entrada    
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

# Función para mostrar la imagen con un recuadro ajustable
def display_image_with_box(image, box):
    img_with_box = image.copy()
    cv2.rectangle(img_with_box, box[0], box[1], color=(0, 255, 0), thickness=2)
    return img_with_box

def main():
    st.title("Clasificación Alada")
    st.header("Sistema Multiclase para la Identificación Aviar en Ibagué")
    st.markdown("""
    ### Información del Proyecto
    El proyecto "Clasificación Alada" es un sistema multiclase diseñado para la identificación de aves en la región de Ibagué, 
    con un enfoque centrado en técnicas de Deep Learning. El objetivo principal es proporcionar una herramienta precisa y eficiente 
    para la clasificación de aves a partir de imágenes.
    
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

    # Se carga el modelo
    model = load_model(MODEL_PATH)

    if choice == "Realizar Predicciones":
        st.subheader("Realizar Predicciones")
        img_file_buffer = st.file_uploader("Carga una imagen", type=["png", "jpg", "jpeg"])

        if img_file_buffer is not None:
            image = np.array(Image.open(img_file_buffer))
            st.image(image, caption="Imagen Original", use_column_width=True)

            # Mostrar la imagen con el recuadro ajustable
            st.subheader("Selecciona la región con el pájaro")
            box = st.image(image, caption="Imagen con Recuadro", use_column_width=True, clamp=True)

        if st.button("Identificar Ave") and 'box' in locals():
            if img_file_buffer is not None:
                predictS = model_prediction(image, model)
                bird_name = names[np.argmax(predictS)]
                st.success(f'El ave es: {bird_name}')

                # Aquí puedes utilizar las coordenadas del recuadro (box) para recortar la región de interés
                # y procesarla para la clasificación si es necesario

                # Ejemplo de cómo obtener las coordenadas del recuadro (box)
                # box_coords = box.coords  # Esto depende de cómo decidas implementar la interacción con el usuario
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
