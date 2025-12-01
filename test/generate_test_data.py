"""
Script para generar datos de prueba para el pipeline CI/CD.
Genera un archivo JSON con muestras MNIST para testing.
"""

import json
import numpy as np


def generate_test_samples(num_samples=20):
    """
    Genera muestras de prueba sintéticas para MNIST.
    
    Args:
        num_samples: Número de muestras a generar
    
    Returns:
        Lista de muestras con formato {image: [...], label: int}
    """
    np.random.seed(42)
    samples = []
    
    for i in range(num_samples):
        # Crear imagen base (28x28)
        image = np.zeros((28, 28), dtype=np.float32)
        
        # Simular un dígito (patrón simple)
        digit = i % 10  # Ciclar entre 0-9
        
        # Agregar ruido y patrón simple según el dígito
        if digit == 0:
            # Círculo
            center_x, center_y = 14, 14
            for x in range(28):
                for y in range(28):
                    dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                    if 6 < dist < 10:
                        image[y, x] = 0.8 + np.random.rand() * 0.2
        
        elif digit == 1:
            # Línea vertical
            image[5:23, 13:15] = 0.8 + np.random.rand(18, 2) * 0.2
        
        elif digit == 2:
            # Forma de "2"
            image[6:9, 8:18] = 0.8 + np.random.rand(3, 10) * 0.2
            image[9:14, 15:18] = 0.8 + np.random.rand(5, 3) * 0.2
            image[14:17, 8:18] = 0.8 + np.random.rand(3, 10) * 0.2
            image[17:22, 8:11] = 0.8 + np.random.rand(5, 3) * 0.2
            image[22:25, 8:18] = 0.8 + np.random.rand(3, 10) * 0.2
        
        elif digit == 3:
            # Forma de "3"
            image[6:9, 8:18] = 0.8 + np.random.rand(3, 10) * 0.2
            image[13:16, 8:18] = 0.8 + np.random.rand(3, 10) * 0.2
            image[21:24, 8:18] = 0.8 + np.random.rand(3, 10) * 0.2
            image[9:21, 15:18] = 0.8 + np.random.rand(12, 3) * 0.2
        
        elif digit == 4:
            # Forma de "4"
            image[6:16, 8:11] = 0.8 + np.random.rand(10, 3) * 0.2
            image[13:16, 8:18] = 0.8 + np.random.rand(3, 10) * 0.2
            image[6:24, 15:18] = 0.8 + np.random.rand(18, 3) * 0.2
        
        elif digit == 5:
            # Forma de "5"
            image[6:9, 8:18] = 0.8 + np.random.rand(3, 10) * 0.2
            image[9:14, 8:11] = 0.8 + np.random.rand(5, 3) * 0.2
            image[14:17, 8:18] = 0.8 + np.random.rand(3, 10) * 0.2
            image[17:22, 15:18] = 0.8 + np.random.rand(5, 3) * 0.2
            image[22:25, 8:18] = 0.8 + np.random.rand(3, 10) * 0.2
        
        elif digit == 6:
            # Forma de "6"
            image[6:24, 8:11] = 0.8 + np.random.rand(18, 3) * 0.2
            image[6:9, 8:18] = 0.8 + np.random.rand(3, 10) * 0.2
            image[14:17, 8:18] = 0.8 + np.random.rand(3, 10) * 0.2
            image[21:24, 8:18] = 0.8 + np.random.rand(3, 10) * 0.2
            image[17:21, 15:18] = 0.8 + np.random.rand(4, 3) * 0.2
        
        elif digit == 7:
            # Forma de "7"
            image[6:9, 8:18] = 0.8 + np.random.rand(3, 10) * 0.2
            image[9:24, 15:18] = 0.8 + np.random.rand(15, 3) * 0.2
        
        elif digit == 8:
            # Forma de "8" (dos círculos)
            image[6:9, 8:18] = 0.8 + np.random.rand(3, 10) * 0.2
            image[14:17, 8:18] = 0.8 + np.random.rand(3, 10) * 0.2
            image[22:25, 8:18] = 0.8 + np.random.rand(3, 10) * 0.2
            image[9:14, 8:11] = 0.8 + np.random.rand(5, 3) * 0.2
            image[9:14, 15:18] = 0.8 + np.random.rand(5, 3) * 0.2
            image[17:22, 8:11] = 0.8 + np.random.rand(5, 3) * 0.2
            image[17:22, 15:18] = 0.8 + np.random.rand(5, 3) * 0.2
        
        elif digit == 9:
            # Forma de "9"
            image[6:9, 8:18] = 0.8 + np.random.rand(3, 10) * 0.2
            image[9:14, 8:11] = 0.8 + np.random.rand(5, 3) * 0.2
            image[9:14, 15:18] = 0.8 + np.random.rand(5, 3) * 0.2
            image[14:17, 8:18] = 0.8 + np.random.rand(3, 10) * 0.2
            image[6:24, 15:18] = 0.8 + np.random.rand(18, 3) * 0.2
        
        # Agregar ruido de fondo
        noise = np.random.rand(28, 28) * 0.05
        image = np.clip(image + noise, 0, 1)
        
        # Convertir a lista (JSON serializable)
        image_list = image.flatten().tolist()
        
        samples.append({
            "image": image_list,
            "label": digit
        })
    
    return samples


def save_test_samples(filename="mnist_test_samples.json", num_samples=20):
    """
    Genera y guarda muestras de prueba en un archivo JSON.
    
    Args:
        filename: Nombre del archivo de salida
        num_samples: Número de muestras a generar
    """
    print(f"Generando {num_samples} muestras de prueba...")
    samples = generate_test_samples(num_samples)
    
    with open(filename, 'w') as f:
        json.dump(samples, f, indent=2)
    
    print(f"✅ {len(samples)} muestras guardadas en {filename}")
    print(f"📊 Tamaño del archivo: {len(json.dumps(samples)) / 1024:.2f} KB")
    
    # Estadísticas
    labels = [s["label"] for s in samples]
    print(f"📈 Distribución de etiquetas:")
    for digit in range(10):
        count = labels.count(digit)
        print(f"   Dígito {digit}: {count} muestras")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generar datos de prueba MNIST sintéticos")
    parser.add_argument(
        "--num-samples",
        type=int,
        default=20,
        help="Número de muestras a generar (default: 20)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="mnist_test_samples.json",
        help="Nombre del archivo de salida (default: mnist_test_samples.json)"
    )
    
    args = parser.parse_args()
    
    save_test_samples(filename=args.output, num_samples=args.num_samples)
    
    print("\n📝 Para subir a Azure Blob Storage:")
    print("   az storage blob upload \\")
    print("     --account-name <STORAGE_ACCOUNT_NAME> \\")
    print("     --account-key <STORAGE_ACCOUNT_KEY> \\")
    print("     --container-name test-data \\")
    print("     --name mnist_test_samples.json \\")
    print(f"     --file ./{args.output}")
