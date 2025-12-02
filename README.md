# MLOps - MNIST Final Project - Entorno Dev

Proyecto final para la materia de MLOps. Esta aplicación es una interfaz web construida con **Streamlit** que utiliza un modelo de Deep Learning (ONNX) para el reconocimiento de dígitos escritos a mano (MNIST).

## Características

- **Reconocimiento en tiempo real**: Predicción instantánea de dígitos del 0 al 9.
- **Doble entrada**: Permite subir imágenes o dibujar directamente en un canvas interactivo.
- **Modelo Optimizado**: Utiliza una versión cuantizada (`int8`) del modelo MNIST para mayor eficiencia y menor latencia.
- **MLOps Pipeline**: Integración y despliegue continuo (CI/CD) automatizado hacia Azure.
- **Monitoreo**: Registro de predicciones y nivel de confianza en Azure Blob Storage.

## Sobre el Modelo

El modelo utilizado es **MNIST-12-int8**, obtenido del repositorio oficial de ONNX Model Zoo. Este modelo es una red neuronal convolucional (CNN) entrenada con el dataset MNIST.

- **Fuente**: [ONNX Models - MNIST](https://github.com/onnx/models/tree/main/validated/vision/classification/mnist)
- **Formato**: ONNX (Open Neural Network Exchange)
- **Entrada**: Tensores de `float32` con forma `(1, 1, 28, 28)`.
- **Preprocesamiento**: Las imágenes se convierten a escala de grises, se redimensionan a 28x28 y se normalizan.

## Arquitectura y Despliegue

El proyecto implementa un flujo de trabajo MLOps completo utilizando **GitHub Actions** y **Microsoft Azure**:

1.  **Control de Versiones**: El código fuente se gestiona en la rama `development`.
2.  **CI/CD Pipeline**: Al hacer push, se dispara un workflow que:
    *   **Test**: Descarga el modelo y datos de prueba de Azure Blob Storage y ejecuta pruebas unitarias (`test/test_model.py`).
    *   **Build**: Construye una imagen Docker que incluye la aplicación y descarga el modelo.
    *   **Push**: Sube la imagen a **Azure Container Registry (ACR)**.
    *   **Deploy**: Despliega una nueva instancia en **Azure Container Instances (ACI)**.

## Estructura del Proyecto

```text
MLOPS--MNIST-FINAL-PROJECT/
├── .github/workflows/   # Pipelines de CI/CD (GitHub Actions)
├── src/                 # Código fuente de la aplicación
│   ├── app.py           # Interfaz principal (Streamlit)
│   └── utils.py         # Lógica de procesamiento de imágenes e inferencia
├── test/                # Pruebas unitarias y de integración
├── Dockerfile           # Configuración para la containerización
├── requirements.txt     # Dependencias de Python
└── README.md            # Documentación del proyecto
```

## Ejecución Local

Si deseas probar la aplicación en tu máquina local:

### Prerrequisitos
- Python 3.9 o superior
- Pip

### Pasos
1. Clonar el repositorio:
   ```bash
   git clone https://github.com/SamuelGutierrez24/MLOPS--MNIST-FINAL-PROJECT.git
   cd MLOPS--MNIST-FINAL-PROJECT
   ```

2. Instalar dependencias:
   ```bash
   pip install -r requirements.txt
   ```

3. Ejecutar la aplicación:
   ```bash
   streamlit run src/app.py
   ```
   *Nota: Si no tienes el modelo `model.onnx` descargado localmente, la aplicación intentará buscarlo en la ruta configurada o fallará si no tiene acceso a Azure.*

## Docker

Para ejecutar la aplicación utilizando Docker (simulando el entorno de producción):

1. Construir la imagen:
   ```bash
   docker build -t mnist-app .
   ```

2. Correr el contenedor:
   ```bash
   docker run -p 8501:8501 mnist-app
   ```
   La aplicación estará disponible en `http://localhost:8501`.

---