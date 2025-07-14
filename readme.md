**Entorno de desarrollo para el -Asistente Virtual Versat-**

***Instrucciones***

1. **Instalar dependencias del sistema**

    *Instalar dependencias del sistema*

        sudo apt update && sudo apt upgrade -y

        sudo apt install software-properties-common -y

        sudo add-apt-repository ppa:deadsnakes/ppa -y

        sudo apt update

2. **Descargar e instalar Ollama (necesita VPN)**

   *Descargar script del instalador de ollama*

        curl -O https://ollama.ai/install.sh

    *Correr el script de instalación*

        sudo bash install.sh

    *Verificar instalacion de Ollama*

        ollama --version

    *Levantar el servidor de Ollama*

    - Si desea ver los logs del servidor

            ollama serve

    - Para levantarlo en segundo plano

            ollama serve >

    *Comprobar que el servidor esta activo*

        curl http://localhost:11434

    La respuesta debe ser ***'Ollama is running'***

    *Descargar los modelos necesarios*

        ollama pull <nombre del modelo>

3. Instalar python 3.12 o superior

        sudo apt install python3.12 -y

        sudo apt install curl git python3-pip

4. **Descargar proyecto del gitlab**

        git clone https://gitlab.azcuba.cu/python/asistente-virtual

5. **Instalar los requerimientos del proyecto**

   *Crear entorno*

        python3 -m venv .venv
   *Activar entorno*

        source ./.venv/bin/activate

   *Instalar requerimientos*

        pip install -r requirements.txt

6. **Levantar el proyecto**

        streamlit run app.py --server.port=8552



*Instalacion utilizando Docker*
