"""
Funciones auxiliares para la aplicación MNIST de Streamlit.
Incluye procesamiento de imágenes y logging en Azure Blob Storage.
"""

import numpy as np
import cv2
from PIL import Image
import os
import logging
from datetime import datetime
from azure.storage.blob import BlobServiceClient

logger = logging.getLogger(__name__)


def process_image(image: Image.Image, is_canvas: bool = False) -> tuple:
    """
    Procesar imagen para hacerla compatible con el modelo ONNX MNIST.
    
    Args:
        image: Imagen PIL a procesar
        is_canvas: Si True, la imagen es del canvas y necesita ajustes especiales
    
    Returns:
        Tupla con:
        - imagen_procesada: numpy array con shape (1, 1, 28, 28) dtype float32
        - imagen_28x28: numpy array de la imagen 28x28 en escala de grises (para visualización)
    """
    try:
        # Convertir a numpy array si es necesario
        if isinstance(image, Image.Image):
            # Si tiene canal alfa, convertir a RGB
            if image.mode == 'RGBA':
                # Crear fondo blanco
                background = Image.new('RGB', image.size, (255, 255, 255))
                background.paste(image, mask=image.split()[3])
                image = background
            
            # Convertir a escala de grises
            if image.mode != 'L':
                image = image.convert('L')
            
            img_array = np.array(image)
        else:
            img_array = image
        
        # Si viene del canvas, convertir a escala de grises si no lo es
        if is_canvas:
            if len(img_array.shape) == 3:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        
        # Redimensionar a 28x28
        img_resized = cv2.resize(img_array, (28, 28), interpolation=cv2.INTER_LINEAR)
        
        # Guardar copia para visualización
        img_display = img_resized.copy()
        
        # Normalizar a [0, 1]
        # El modelo MNIST espera: fondo NEGRO (0.0) y dígito BLANCO (1.0)
        img_normalized = img_resized.astype('float32') / 255.0
        
        # Reshape a (1, 1, 28, 28) para batch_size=1, channels=1
        img_final = img_normalized.reshape(1, 1, 28, 28).astype('float32')
        
        logger.info(f"Imagen procesada exitosamente. Shape: {img_final.shape}, dtype: {img_final.dtype}")
        logger.info(f"Rango de valores: min={img_final.min():.3f}, max={img_final.max():.3f}, mean={img_final.mean():.3f}")
        
        return img_final, img_display
    
    except Exception as e:
        logger.error(f"Error procesando imagen: {str(e)}")
        raise


def log_prediction_to_azure(
    timestamp: str,
    prediction: int,
    confidence: float,
    environment: str = "dev"
) -> bool:
    """
    Guardar predicción en Azure Blob Storage.
    
    Args:
        timestamp: Timestamp ISO format de la predicción
        prediction: Dígito predicho (0-9)
        confidence: Confianza de la predicción (0-1)
        environment: Entorno (dev o prod)
    
    Returns:
        True si se guardó exitosamente, False en caso contrario
    """
    try:
        connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        
        if not connection_string:
            logger.warning("AZURE_STORAGE_CONNECTION_STRING no configurado. Log local solamente.")
            _log_prediction_locally(timestamp, prediction, confidence, environment)
            return True
        
        # Crear cliente de blob
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        
        # Nombre del contenedor y blob
        container_name = "predicciones"
        blob_name = f"predicciones_{environment}.txt"
        
        # Obtener referencia al contenedor (crear si no existe)
        try:
            blob_service_client.create_container(name=container_name)
            logger.info(f"Contenedor '{container_name}' creado.")
        except Exception as e:
            # El contenedor probablemente ya existe
            logger.debug(f"Contenedor ya existe o error: {str(e)}")
        
        # Preparar el contenido a guardar
        log_entry = f"{timestamp} | Predicción: {prediction} | Confianza: {confidence:.2%} | Entorno: {environment}\n"
        
        # Obtener blob actual
        blob_client = blob_service_client.get_blob_client(
            container=container_name,
            blob=blob_name
        )
        
        # Leer contenido existente
        try:
            existing_content = blob_client.download_blob().readall().decode('utf-8')
        except:
            existing_content = ""
        
        # Agregar nueva entrada
        new_content = existing_content + log_entry
        
        # Subir contenido actualizado
        blob_client.upload_blob(new_content, overwrite=True)
        
        logger.info(f"Predicción guardada en Azure Blob Storage: {blob_name}")
        return True
    
    except Exception as e:
        logger.error(f"Error guardando predicción en Azure: {str(e)}")
        # Intentar guardar localmente como fallback
        _log_prediction_locally(timestamp, prediction, confidence, environment)
        return False


def _log_prediction_locally(
    timestamp: str,
    prediction: int,
    confidence: float,
    environment: str
) -> None:
    """
    Guardar predicción en archivo local (fallback).
    
    Args:
        timestamp: Timestamp ISO format
        prediction: Dígito predicho
        confidence: Confianza
        environment: Entorno (dev o prod)
    """
    try:
        filename = f"predicciones_{environment}.txt"
        log_entry = f"{timestamp} | Predicción: {prediction} | Confianza: {confidence:.2%} | Entorno: {environment}\n"
        
        with open(filename, 'a') as f:
            f.write(log_entry)
        
        logger.info(f"Predicción guardada localmente: {filename}")
    except Exception as e:
        logger.error(f"Error guardando predicción localmente: {str(e)}")


def get_latest_predictions(limit: int = 10) -> list:
    """
    Obtener las últimas predicciones registradas.
    
    Args:
        limit: Cantidad máxima de predicciones a retornar
    
    Returns:
        Lista con las últimas predicciones
    """
    try:
        environment = os.getenv("ENVIRONMENT", "dev")
        connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        
        predictions = []
        
        if connection_string:
            try:
                blob_service_client = BlobServiceClient.from_connection_string(connection_string)
                blob_client = blob_service_client.get_blob_client(
                    container="mnist-predictions",
                    blob=f"predicciones_{environment}.txt"
                )
                
                content = blob_client.download_blob().readall().decode('utf-8')
                lines = content.strip().split('\n')
                predictions = lines[-limit:] if lines else []
                
                logger.info(f"Se obtuvieron {len(predictions)} predicciones de Azure")
            except Exception as e:
                logger.debug(f"Error obtuviendo predicciones de Azure: {str(e)}")
                predictions = _get_local_predictions(limit, environment)
        else:
            predictions = _get_local_predictions(limit, environment)
        
        return predictions
    
    except Exception as e:
        logger.error(f"Error obteniendo predicciones: {str(e)}")
        return []


def _get_local_predictions(limit: int, environment: str) -> list:
    """
    Obtener predicciones desde archivo local.
    
    Args:
        limit: Cantidad máxima
        environment: Entorno (dev o prod)
    
    Returns:
        Lista con predicciones locales
    """
    try:
        filename = f"predicciones_{environment}.txt"
        
        if not os.path.exists(filename):
            return []
        
        with open(filename, 'r') as f:
            lines = f.readlines()
        
        # Obtener las últimas 'limit' líneas
        return [line.strip() for line in lines[-limit:]]
    
    except Exception as e:
        logger.error(f"Error leyendo predicciones locales: {str(e)}")
        return []


def validate_image_format(file_extension: str) -> bool:
    """
    Validar que la extensión de archivo sea soportada.
    
    Args:
        file_extension: Extensión del archivo
    
    Returns:
        True si es válida, False en caso contrario
    """
    valid_extensions = ['.png', '.jpg', '.jpeg']
    return file_extension.lower() in valid_extensions


def get_model_info() -> dict:
    """
    Obtener información del modelo.
    
    Returns:
        Diccionario con información del modelo
    """
    return {
        "nombre": "MNIST Int8 ONNX",
        "input_shape": (1, 1, 28, 28),
        "input_type": "float32",
        "output_shape": (1, 10),
        "output_labels": [str(i) for i in range(10)],
        "version": "1.0",
        "descripcion": "Modelo ONNX para reconocimiento de dígitos MNIST"
    }
