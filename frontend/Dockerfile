FROM python:3.9-slim

# Establece el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copia las dependencias del frontend
COPY requirements.txt requirements.txt

# Instala las dependencias
RUN pip install -r requirements.txt

# Copia el código del frontend al contenedor
COPY . .

# Expone el puerto en el que correrá la interfaz de usuario
EXPOSE 8501

# Comando para ejecutar la aplicación de Streamlit con los argumentos requeridos
CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0", "--server.port", "8501"]