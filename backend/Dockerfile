# Usa una imagen ligera de Python como base
FROM python:3.9-slim

# Establece el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copia las dependencias del backend
COPY requirements.txt requirements.txt

# Instala las dependencias
RUN pip install -r requirements.txt

# Copia el código del backend al contenedor
COPY . .

# Expone el puerto en el que correrá el servicio
EXPOSE 5000

# Comando para ejecutar el servicio de FastAPI
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5000"]
