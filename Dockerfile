# ============================================
# Dockerfile para Aplicación MNIST con ONNX
# ============================================
# Imagen optimizada para despliegue en Azure Container Instances

# Usar imagen base oficial de Python (slim = más liviana)
FROM python:3.9-slim

# Metadata
LABEL maintainer="MLOps Project"
LABEL description="MNIST Digit Recognition with ONNX and Streamlit"

# Argumentos de build (se pasan desde GitHub Actions)
ARG MODEL_URL
ARG BUILD_DATE
ARG VERSION=1.0

# Variables de entorno
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MODEL_PATH=/app/model.onnx

# Establecer directorio de trabajo
WORKDIR /app

# ============================================
# PASO 1: Instalar dependencias del sistema
# ============================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ============================================
# PASO 2: Copiar e instalar dependencias Python
# ============================================
# Copiar requirements.txt primero (para aprovechar cache de Docker)
COPY requirements.txt .

# Instalar dependencias de Python
RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir -r requirements.txt

# ============================================
# PASO 3: Descargar modelo ONNX desde Azure Blob
# ============================================
# El modelo NO está en el repositorio, se descarga durante el build
RUN if [ -z "$MODEL_URL" ]; then \
        echo "❌ ERROR: MODEL_URL no está definido"; \
        exit 1; \
    fi && \
    echo "📥 Descargando modelo desde: $MODEL_URL" && \
    wget --progress=bar:force:noscroll -O ${MODEL_PATH} "${MODEL_URL}" && \
    echo "✅ Modelo descargado exitosamente" && \
    ls -lh ${MODEL_PATH}

# Verificar que el modelo se descargó correctamente
RUN if [ ! -f ${MODEL_PATH} ]; then \
        echo "❌ ERROR: El modelo no se descargó correctamente"; \
        exit 1; \
    fi && \
    echo "✅ Modelo verificado: $(du -h ${MODEL_PATH})"

# ============================================
# PASO 4: Copiar código de la aplicación
# ============================================
COPY src/app.py .
COPY src/utils.py* ./

# ============================================
# PASO 5: Limpiar para reducir tamaño de imagen
# ============================================
RUN apt-get purge -y --auto-remove wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# ============================================
# PASO 6: Crear usuario no-root (seguridad)
# ============================================
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

USER appuser

# ============================================
# PASO 7: Exponer puerto de Streamlit
# ============================================
EXPOSE 8501

# ============================================
# PASO 8: Healthcheck para Container Instances
# ============================================
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# ============================================
# PASO 9: Comando de inicio
# ============================================
CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.serverAddress=0.0.0.0", \
     "--browser.gatherUsageStats=false"]