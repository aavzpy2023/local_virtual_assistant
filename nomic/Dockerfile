FROM python:3.9-slim

WORKDIR /app

# Copiar dependencias
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copiar el código
COPY . .

# Comando para ejecutar el servicio
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
