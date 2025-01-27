from keras.models import load_model
from keras.applications.imagenet_utils import preprocess_input
import numpy as np
import streamlit as st
from PIL import Image
from skimage.transform import resize
import pandas as pd
import os


html_content = """
<div style="width: 100%; clear: both; display: flex; align-items: center; justify-content: space-between;">
    <div style="width: 50%; display: flex; justify-content: flex-start;">
        <img src="https://www.bing.com/images/blob?bcid=suFss51W2wgIpA" style="width: 100%; max-width: 500px; height: auto;">
    </div>
    <div style="width: 50%; text-align: right; padding-left: 0px;">
        <p style="margin: 0px; font-weight: bold;">Laboratorio de Tecnologías Emergentes</p>
        <p style="margin: 0px;">Universidad Cooperativa de Colombia, Campus Ibagué-Espinal</p>
        <p style="margin: 0px;">Facultad de Ingeniería</p>
        <p style="margin: 0px;">Programa de Ingeniería de Sistemas</p>
        <p style="margin: 0px;">2025</p>
    </div>
</div>
"""


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

# Se recibe la imagen y el modelo, devuelve la predicción
def model_prediction(img, model):
    img_resize = resize(img, (width_shape, height_shape), anti_aliasing=True)
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
    st.markdown(html_content, unsafe_allow_html=True)
    # Se intenta cargar el modelo
    try:
        model = load_model(MODEL_PATH)
        st.success("Modelo cargado correctamente")
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        return
    
    st.image("banner2.jpg", use_container_width=True)
    
    st.title("Tolima Birds AI")
    st.header("Sistema de Identificación Aviar Basado en Inteligencia Artificial")

    # Menú
    menu = ["Información del Proyecto", "Realizar Predicciones", "Listar Aves Entrenadas", "Agradecimientos", "Trabajos Futuros"]
    choice = st.sidebar.selectbox("Selecciona una opción", menu)

    if choice == "Realizar Predicciones":
        st.subheader("Realizar Predicciones")
        st.markdown("""
        En esta sección, podrás cargar una imagen de un ave para identificar su especie utilizando el sistema Tolima Birds AI. Una vez cargada la imagen, el sistema procesará la información y mostrará los resultados, incluyendo el nombre de la especie, detalles relevantes sobre su hábitat y características principales, además de imágenes adicionales relacionadas con el ave identificada. Este proceso está diseñado para ofrecer una experiencia intuitiva y educativa, apoyando la investigación y conservación de la biodiversidad aviar en el Tolima.
        """)
        img_file_buffer = st.file_uploader("Carga una imagen", type=["png", "jpg", "jpeg"])
        
        if img_file_buffer is not None:
            image = np.array(Image.open(img_file_buffer))
            
            if image.ndim == 2:  # Si la imagen es en escala de grises, conviértela a RGB
                image = np.stack([image]*3, axis=-1)
            elif image.shape[2] == 4:  # Si la imagen tiene un canal alfa, elimínalo
                image = image[:, :, :3]
            
            st.image(image, caption="Imagen", use_container_width=True)
        
        if st.button("Identificar Ave"):
            if img_file_buffer is not None:
                try:
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
                            cols[i % 3].image(img, use_container_width=True)
                    else:
                        st.warning("No se encontraron imágenes adicionales del ave en la galería.")
                except ValueError as e:
                    st.error(f"Error de predicción: {e}")
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

            col1, col2, col3 = st.columns(3)

            for j, bird in enumerate(bird_row):
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

    elif choice == "Información del Proyecto":
        st.subheader("Información del Proyecto")
        st.markdown("""
    
    Tolima Birds AI es un sistema innovador diseñado para la identificación y clasificación automatizada de aves en el departamento del Tolima, empleando técnicas avanzadas de inteligencia artificial. Este proyecto busca apoyar la conservación de la biodiversidad aviar de la región, facilitando la identificación precisa de especies mediante el análisis de imágenes y datos.

    La iniciativa combina el poder de las redes neuronales convolucionales con un enfoque en la diversidad ecológica local, permitiendo a investigadores, estudiantes y conservacionistas acceder a una herramienta eficiente y de fácil uso para el monitoreo y estudio de las aves de Tolima. Además, el sistema contribuye a la generación de conocimiento científico y fomenta la educación ambiental en la comunidad, promoviendo el cuidado del entorno natural.

    Este proyecto, desarrollado con una metodología interdisciplinaria, busca posicionarse como una referencia en la integración de tecnología de punta en la conservación de ecosistemas regionales.
    """)
                    
    
        st.subheader("**Investigadores:**")
        
        st.write("Oscar Augusto Diaz Triana")
        st.write("Fernando Gutierrez Portela")
        st.write("Universidad Cooperativa de Colombia - Campus Ibagué-Espinal") 
        st.write("Programa de Ingeniería de Sistemas")
        st.write("2024")
    
        
    elif choice == "Agradecimientos":
        st.subheader("Agradecimientos")
        st.markdown("""
        Este trabajo ha sido posible gracias al apoyo de la Universidad Cooperativa de Colombia, así como al respaldo del Comité Nacional para el Desarrollo de la Investigación (CONADI), cuyo compromiso con la promoción de la investigación ha sido fundamental para el desarrollo de este proyecto.
        """)
    
    elif choice == "Trabajos Futuros":
        st.subheader("Trabajos Futuros: Segmentación Semántica para Evaluación de Imágenes de Aves")

        st.write("Uno de los próximos pasos es incorporar segmentación semántica para determinar si una imagen proporcionada corresponde efectivamente a un ave. Esta técnica permitirá mejorar la precisión del sistema al enfocarse no solo en la clasificación de especies, sino también en la identificación precisa de la presencia de aves en las imágenes. Esto es crucial para aplicaciones que requieren una detección precisa del objeto de interés.")

        st.write("La segmentación semántica puede ayudar a distinguir entre aves y otros objetos en entornos complejos, lo que es especialmente útil en aplicaciones de monitoreo de fauna, conservación y estudios ecológicos.")

        st.write("Este trabajo futuro también implicará la implementación de una arquitectura adicional de redes neuronales convolucionales adaptada específicamente para la segmentación semántica de aves en imágenes de campo, incorporando técnicas avanzadas como U-Net o Mask R-CNN.")

        st.write("La integración de esta funcionalidad ampliará las capacidades del sistema actual, facilitando su uso en diversas aplicaciones de investigación y conservación.")

st.markdown("""
    <hr>
    <p style="text-align: center; font-size: 14px;">
        Universidad Cooperativa de Colombia - Campus Ibagué-Espinal <br>
        Facultad de Ingeniería 2025
    </p>
    """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()


