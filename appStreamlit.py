from keras.models import load_model
from keras.applications.imagenet_utils import preprocess_input
import numpy as np
import streamlit as st
from PIL import Image
from skimage.transform import resize
import pandas as pd
import os

# Path del modelo preentrenado
MODEL_PATH = 'models/optimizado.keras'

# Path del archivo Excel
EXCEL_PATH = 'aves.xlsx'

# Dimensiones de las imágenes de entrada    
width_shape = 224
height_shape = 224

# Clases
names = ['CATHARTES AURA', 'COEREBA FLAVEOLA', 'COLUMBA LIVIA', 'CORAGYPS ATRATUS', 'CROTOPHAGA SULCIROSTRIS', 'CYANOCORAX YNCAS',
         'EGRETTA THULA', 'FALCO PEREGRINUS', 'FALCO SPARVERIUS', 'HIRUNDO RUSTICA', 'PANDION HALIAETUS', 'PILHERODIUS PILEATUS',
         'PITANGUS SULPHURATUS', 'PYRRHOMYIAS CINNAMOMEUS', 'RYNCHOPS NIGER', 'SETOPHAGA FUSCA', 'SYNALLAXIS AZARAE', 'TYRANNUS MELANCHOLICUS']

# Extensiones permitidas
ALLOWED_EXTENSIONS = ('.png', '.jpg', '.jpeg')

# Se carga el modelo al inicio
@st.cache(allow_output_mutation=True)
def load_keras_model(model_path):
    try:
        model = load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        return None

model = load_keras_model(MODEL_PATH)

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
        return bird_info.iloc[0]  # Selecciona la primera fila (debería ser única)
    else:
        return None

def load_bird_images(bird_name):
    bird_dir = os.path.join('datasetpreprocesado/test', bird_name.replace(' ', '_'))
    
    bird_name_buscar = bird_name.replace(' ', '+')
    st.markdown(f"[Ver más Información](https://www.google.com/search?q={bird_name_buscar})")
    
    if os.path.exists(bird_dir):
        images = []
        for img_file in os.listdir(bird_dir):
            if img_file.endswith(ALLOWED_EXTENSIONS):
                img_path = os.path.join(bird_dir, img_file)
                images.append(img_path)
        return images
    else:
        return []

def main():
    # Fondo o banner de la aplicación
    st.image("banner2.jpg", use_column_width=True)
    
    st.title("Clasificación Alada")
    st.header("Sistema Multiclase para la Identificación Aviar en Ibagué")
    st.markdown("""
    ### Información del Proyecto
    El proyecto "Clasificación Alada" es un sistema multiclase diseñado para la identificación de aves en la región de Ibagué, 
    con un enfoque centrado en técnicas de Deep Learning. El objetivo principal es proporcionar una herramienta precisa y eficiente 
    para la clasificación de aves a partir de imágenes. Este trabajo fue apoyado  por el Ministerio de Tecnologías de la Información y las Comunicaciones de Colombia y la Universidad Cooperativa de Colombia, Campus Ibagué - Espinal.
    
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

    if model is None:
        st.error("No se pudo cargar el modelo. Por favor, revisa el path y el archivo del modelo.")
        return

    if choice == "Realizar Predicciones":
        st.subheader("Realizar Predicciones")
        img_file_buffer = st.file_uploader("Carga una imagen", type=["png", "jpg", "jpeg"])
        
        if img_file_buffer is not None:
            image = np.array(Image.open(img_file_buffer))    
            st.image(image, caption="Imagen", use_column_width=True)
        
        if st.button("Identificar Ave"):
            if img_file_buffer is not None:
                preds = model_prediction(image, model)
                bird_name = names[np.argmax(preds)]
                confidence = np.max(preds)
                st.success(f'El ave es: {bird_name} con una precisión del {confidence:.2%}')

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
            {"name": "PYRRHOMYIAS+CINNAMOMEUS", "image": "static/imagen/PYRRHOMYIAS CINNAMOMEUS_4.jpg"},
            {"name": "RYNCHOPS+NIGER", "image": "static/imagen/RYNCHOPS NIGER_15.jpeg"},
            {"name": "SETOPHAGA+FUSCA", "image": "static/imagen/SETOPHAGA FUSCA_2.jpg"},
            {"name": "SYNALLAXIS+AZARAE", "image": "static/imagen/SYNALLAXIS AZARAE_9.jpeg"},
            {"name": "TYRANNUS+MELANCHOLICUS", "image": "static/imagen/TYRANNUS MELANCHOLICUS_17.jpeg"}
        ]
        for bird in birds_info:
            bird_image = Image.open(bird["image"])
            st.image(bird_image, caption=bird["name"], use_column_width=True)
            st.markdown(f"[Ver más Información](https://www.google.com/search?q={bird['name']})")
            
    elif choice == "Agradecimientos":
        st.header("Agradecimientos")
        st.markdown("""
        Este proyecto no habría sido posible sin el apoyo y la colaboración de muchas personas y organizaciones. 
        Queremos expresar nuestro más sincero agradecimiento a:
        
        - **Universitat Oberta de Catalunya**: Por proporcionar una plataforma educativa excepcional y apoyo académico.
        - **Ministerio de Tecnologías de la Información y las Comunicaciones de Colombia**: Por su apoyo financiero y logístico.
        - **Universidad Cooperativa de Colombia, Campus Ibagué - Espinal**: Por su infraestructura y recursos para la investigación.
        - **Bernat Bas Pujols y Pablo Fernandez Blanco**: Por su invaluable orientación y asesoría.
        - **Albert Solé**: Por su liderazgo y enseñanza en la asignatura.
        """)

if __name__ == '__main__':
    main()
