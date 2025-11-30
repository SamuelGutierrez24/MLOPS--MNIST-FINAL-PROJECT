"""
Tests unitarios para el modelo MNIST ONNX.
Requisitos del taller:
1. Probar que el modelo responde con datos de entrada definidos
2. Probar que no existe un cambio significativo en alguna métrica definida
"""

import unittest
import numpy as np
import onnxruntime as rt
import os
from azure.storage.blob import BlobServiceClient
import tempfile
import json


class TestMNISTModel(unittest.TestCase):
    """Tests para validar el modelo MNIST ONNX"""
    
    @classmethod
    def setUpClass(cls):
        """Configuración inicial - descargar modelo y datos de prueba"""
        print("\n" + "="*60)
        print("Iniciando pruebas del modelo MNIST")
        print("="*60)
        
        # Descargar modelo desde Azure Blob Storage
        cls.model_path = cls._download_model_from_azure()
        
        # Descargar datos de prueba desde Azure Blob Storage
        cls.test_data = cls._download_test_data_from_azure()
        
        # Cargar sesión ONNX
        cls.session = rt.InferenceSession(cls.model_path)
        cls.input_name = cls.session.get_inputs()[0].name
        cls.output_name = cls.session.get_outputs()[0].name
        
        print(f"Modelo cargado: {cls.model_path}")
        print(f"Datos de prueba cargados: {len(cls.test_data)} muestras")
    
    @staticmethod
    def _download_model_from_azure():
        """Descargar modelo ONNX desde Azure Blob Storage"""
        try:
            storage_account_name = os.getenv("AZURE_STORAGE_ACCOUNT_NAME")
            storage_account_key = os.getenv("AZURE_STORAGE_ACCOUNT_KEY")
            
            if not storage_account_name or not storage_account_key:
                raise ValueError("Variables de entorno de Azure Storage no configuradas")
            
            connection_string = (
                f"DefaultEndpointsProtocol=https;"
                f"AccountName={storage_account_name};"
                f"AccountKey={storage_account_key};"
                f"EndpointSuffix=core.windows.net"
            )
            
            blob_service_client = BlobServiceClient.from_connection_string(connection_string)
            
            # Descargar modelo
            container_name = "modelos"
            blob_name = "mnist-12-int8.onnx"
            
            blob_client = blob_service_client.get_blob_client(
                container=container_name,
                blob=blob_name
            )
            
            # Guardar en archivo temporal
            model_path = tempfile.NamedTemporaryFile(delete=False, suffix=".onnx").name
            
            with open(model_path, "wb") as f:
                f.write(blob_client.download_blob().readall())
            
            print(f"Modelo descargado desde Azure: {container_name}/{blob_name}")
            return model_path
        
        except Exception as e:
            print(f"Error descargando modelo: {str(e)}")
            raise
    
    @staticmethod
    def _download_test_data_from_azure():
        """Descargar datos de prueba desde Azure Blob Storage"""
        try:
            storage_account_name = os.getenv("AZURE_STORAGE_ACCOUNT_NAME")
            storage_account_key = os.getenv("AZURE_STORAGE_ACCOUNT_KEY")
            
            connection_string = (
                f"DefaultEndpointsProtocol=https;"
                f"AccountName={storage_account_name};"
                f"AccountKey={storage_account_key};"
                f"EndpointSuffix=core.windows.net"
            )
            
            blob_service_client = BlobServiceClient.from_connection_string(connection_string)
            
            # Descargar datos de prueba
            container_name = "test-data"
            blob_name = "mnist_test_samples.json"
            
            blob_client = blob_service_client.get_blob_client(
                container=container_name,
                blob=blob_name
            )
            
            test_data_json = blob_client.download_blob().readall().decode('utf-8')
            test_data = json.loads(test_data_json)
            
            print(f"Datos de prueba descargados desde Azure: {container_name}/{blob_name}")
            return test_data
        
        except Exception as e:
            print(f"Error descargando datos de prueba: {str(e)}")
            raise
    
    def test_01_model_loads_successfully(self):
        """Test: Verificar que el modelo se carga correctamente"""
        print("\nTest 1: Verificando carga del modelo...")
        
        self.assertIsNotNone(self.session, "El modelo no se cargó correctamente")
        self.assertEqual(len(self.session.get_inputs()), 1, "El modelo debe tener 1 entrada")
        self.assertEqual(len(self.session.get_outputs()), 1, "El modelo debe tener 1 salida")
        
        print("Test 1 PASADO: Modelo cargado correctamente")
    
    def test_02_model_input_shape(self):
        """Test: Verificar que el modelo tenga la forma de entrada correcta"""
        print("\nTest 2: Verificando forma de entrada del modelo...")
        
        input_shape = self.session.get_inputs()[0].shape
        expected_shape = [1, 1, 28, 28]
        
        # Comparar dimensiones (ignorar batch size dinámico)
        self.assertEqual(input_shape[1:], expected_shape[1:], 
                        f"La forma de entrada debe ser {expected_shape}, pero es {input_shape}")
        
        print(f"Test 2 PASADO: Forma de entrada correcta: {input_shape}")
    
    def test_03_model_responds_with_defined_input(self):
        """
        Test REQUERIDO 1: Probar que el modelo responde con datos de entrada definidos
        """
        print("\nTest 3: Probando respuesta del modelo con datos de entrada definidos...")
        
        # Crear imagen de prueba (imagen de un "5" sintético)
        test_image = np.zeros((1, 1, 28, 28), dtype=np.float32)
        test_image[0, 0, 10:18, 10:20] = 1.0  # Forma simple que simula un dígito
        
        # Realizar inferencia
        result = self.session.run([self.output_name], {self.input_name: test_image})
        output = result[0][0]
        
        # Validaciones básicas
        self.assertEqual(len(output), 10, "Debe haber 10 salidas (una por dígito)")
        self.assertIsInstance(output, (np.ndarray, list), "La salida debe ser un array")
        
        # El modelo devuelve logits o valores crudos, no probabilidades
        # Verificar que hay variación en las salidas (no todas son iguales)
        self.assertGreater(np.std(output), 0.0, "Las salidas deben tener variación")
        
        # Obtener predicción (la clase con mayor valor)
        predicted_digit = np.argmax(output)
        max_value = output[predicted_digit]
        
        # Verificar que la predicción es válida (0-9)
        self.assertIn(predicted_digit, range(10), "La predicción debe estar entre 0-9")
        
        print(f"   Predicción: {predicted_digit} (valor: {max_value:.4f})")
        print(f"   Rango de salidas: min={np.min(output):.4f}, max={np.max(output):.4f}")
        print(f"Test 3 PASADO: El modelo responde correctamente a datos de entrada definidos")
    
    def test_04_model_accuracy_threshold(self):
        """
        Test REQUERIDO 2: Probar que no existe un cambio significativo en alguna métrica definida
        
        Métrica: Precisión (accuracy) >= 85% en datos de prueba
        """
        print("\nTest 4: Verificando precisión del modelo con datos de prueba...")
        
        # Umbral de precisión mínimo aceptable
        ACCURACY_THRESHOLD = 0.85
        
        correct_predictions = 0
        total_predictions = 0
        
        for sample in self.test_data:
            # Convertir imagen de prueba a formato correcto
            image = np.array(sample["image"], dtype=np.float32).reshape(1, 1, 28, 28)
            true_label = sample["label"]
            
            # Realizar inferencia
            result = self.session.run([self.output_name], {self.input_name: image})
            output = result[0][0]  # Logits o valores crudos
            predicted_digit = np.argmax(output)  # Clase con mayor valor
            
            if predicted_digit == true_label:
                correct_predictions += 1
            total_predictions += 1
        
        # Calcular precisión
        accuracy = correct_predictions / total_predictions
        
        print(f"   Predicciones correctas: {correct_predictions}/{total_predictions}")
        print(f"   Precisión del modelo: {accuracy:.2%}")
        print(f"   Umbral mínimo requerido: {ACCURACY_THRESHOLD:.2%}")
        
        # Validar que la precisión no sea menor al umbral
        self.assertGreaterEqual(
            accuracy, 
            ACCURACY_THRESHOLD,
            f"La precisión del modelo ({accuracy:.2%}) es menor al umbral requerido ({ACCURACY_THRESHOLD:.2%})"
        )
        
        print(f"Test 4 PASADO: Precisión del modelo ({accuracy:.2%}) cumple el umbral ({ACCURACY_THRESHOLD:.2%})")
    
    def test_05_model_all_test_samples(self):
        """Test: Verificar que el modelo puede procesar todas las muestras de prueba"""
        print(f"\nTest 5: Procesando {len(self.test_data)} muestras de prueba...")
        
        for i, sample in enumerate(self.test_data):
            with self.subTest(i=i):
                image = np.array(sample["image"], dtype=np.float32).reshape(1, 1, 28, 28)
                true_label = sample["label"]
                
                # Realizar inferencia
                result = self.session.run([self.output_name], {self.input_name: image})
                output = result[0][0]
                predicted_digit = np.argmax(output)
                
                # Verificar que la predicción esté en el rango válido
                self.assertIn(predicted_digit, range(10), 
                            f"Predicción {predicted_digit} no es un dígito válido (0-9)")
        
        print(f"Test 5 PASADO: Todas las {len(self.test_data)} muestras procesadas exitosamente")
    
    def test_06_model_confidence_reasonable(self):
        """Test: Verificar que la separación entre clases sea razonable"""
        print("\nTest 6: Verificando separación entre clases del modelo...")
        
        MIN_MARGIN = 0.1  # Margen mínimo entre la mejor y segunda mejor predicción
        
        margins = []
        
        for sample in self.test_data:
            image = np.array(sample["image"], dtype=np.float32).reshape(1, 1, 28, 28)
            
            result = self.session.run([self.output_name], {self.input_name: image})
            output = result[0][0]
            
            # Obtener los dos valores más altos
            sorted_values = np.sort(output)[::-1]
            margin = sorted_values[0] - sorted_values[1]  # Diferencia entre primero y segundo
            margins.append(margin)
        
        avg_margin = np.mean(margins)
        
        print(f"   Margen promedio: {avg_margin:.4f}")
        print(f"   Margen mínimo: {np.min(margins):.4f}")
        print(f"   Margen máximo: {np.max(margins):.4f}")
        
        self.assertGreaterEqual(
            avg_margin,
            MIN_MARGIN,
            f"El margen promedio ({avg_margin:.4f}) es menor al umbral ({MIN_MARGIN:.4f})"
        )
        
        print(f"Test 6 PASADO: Margen promedio ({avg_margin:.4f}) es adecuado")
    
    @classmethod
    def tearDownClass(cls):
        """Limpieza después de las pruebas"""
        print("\n" + "="*60)
        print("Limpiando archivos temporales...")
        
        if hasattr(cls, 'model_path') and os.path.exists(cls.model_path):
            os.remove(cls.model_path)
            print(f"Modelo temporal eliminado: {cls.model_path}")
        
        print("Pruebas completadas exitosamente")
        print("="*60)


if __name__ == "__main__":
    # Configurar runner de tests con verbosidad
    unittest.main(verbosity=2)
