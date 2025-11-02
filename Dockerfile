# Dockerfile para ML ClashRoyale con Airflow
# Imagen base: Python 3.11 (compatible con Airflow 3.1.0)
FROM python:3.11-slim

# Metadatos
LABEL maintainer="ML ClashRoyale Team"
LABEL description="Contenedor con Kedro, Airflow y modelos ML para Clash Royale"

# Variables de entorno
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    AIRFLOW_HOME=/opt/airflow \
    AIRFLOW__CORE__LOAD_EXAMPLES=False \
    AIRFLOW__CORE__DAGS_FOLDER=/opt/airflow/dags \
    AIRFLOW__CORE__EXECUTOR=LocalExecutor \
    AIRFLOW__API__AUTH_BACKENDS=airflow.auth.backends.simple_auth_manager \
    AIRFLOW__WEBSERVER__SECRET_KEY=clashroyale_ml_secret_key_change_in_production

# Instalar dependencias del sistema necesarias para Airflow, ML y compilación
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libffi-dev \
    libssl-dev \
    libpq-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Crear usuario no-root para Airflow
RUN useradd -m -u 50000 airflow && \
    mkdir -p ${AIRFLOW_HOME} && \
    chown -R airflow:airflow ${AIRFLOW_HOME}

# Establecer directorio de trabajo
WORKDIR /app

# Copiar requirements.txt primero (para cacheo de capas)
COPY requirements.txt .

# Instalar dependencias de Python
# Primero instalar Airflow con constraints para evitar conflictos
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    AIRFLOW_VERSION=3.1.0 && \
    CONSTRAINT_URL="https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-3.11.txt" && \
    pip install "apache-airflow==${AIRFLOW_VERSION}" --constraint "${CONSTRAINT_URL}" && \
    pip install --no-cache-dir -r requirements.txt

# Copiar código del proyecto
COPY . /app/

# Crear estructura de directorios de Airflow
RUN mkdir -p ${AIRFLOW_HOME}/dags ${AIRFLOW_HOME}/logs ${AIRFLOW_HOME}/plugins

# Copiar y configurar script de entrada
COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

# Cambiar propietario de archivos antes de cambiar de usuario
RUN chown -R airflow:airflow ${AIRFLOW_HOME} /app /docker-entrypoint.sh

# Cambiar a usuario airflow
USER airflow

# Exponer puerto de Airflow
EXPOSE 8080

ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["standalone"]

