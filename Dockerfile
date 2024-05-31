```Dockerfile
# Usa una imagen base oficial de Python
FROM python3.9-slim

# Establece el directorio de trabajo en el contenedor
WORKDIR app

# Copia los archivos de requisitos y el código en el contenedor
COPY requirements.txt requirements.txt
COPY . .

# Instala las dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Expone el puerto en el que correrá la aplicación Flask
EXPOSE 5000

# Define el comando por defecto para correr la aplicación
CMD [gunicorn, -w, 4, -b, 0.0.0.05000, appapp]
```