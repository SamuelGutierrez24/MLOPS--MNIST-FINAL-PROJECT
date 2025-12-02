import streamlit as st
import numpy as np
import onnxruntime as rt
from PIL import Image
import cv2
from datetime import datetime
import os
import plotly.graph_objects as go
from streamlit_drawable_canvas import st_canvas
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Importar funciones auxiliares
try:
    # Intenta importar desde src (cuando se ejecuta localmente)
    from src.utils import (
        process_image,
        log_prediction_to_azure,
        get_latest_predictions
    )
except ModuleNotFoundError:
    # Importa directamente (cuando se ejecuta en Docker)
    from utils import (
        process_image,
        log_prediction_to_azure,
        get_latest_predictions
    )

# Configurar página
st.set_page_config(
    page_title="MNIST - Reconocimiento de Dígitos",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Variables de configuración
MODEL_PATH = "/app/model.onnx"
CANVAS_WIDTH = 300
CANVAS_HEIGHT = 300

# Estilos CSS personalizados
st.markdown("""
    <style>
    .digit-display {
        font-size: 120px;
        text-align: center;
        font-weight: bold;
        padding: 20px;
        border-radius: 10px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin: 20px 0;
    }
    .confidence-display {
        font-size: 24px;
        text-align: center;
        color: #667eea;
        font-weight: bold;
    }
    .error-box {
        padding: 15px;
        border-radius: 5px;
        background-color: #ffcccc;
        border-left: 4px solid #ff0000;
        color: #cc0000;
    }
    .success-box {
        padding: 15px;
        border-radius: 5px;
        background-color: #ccffcc;
        border-left: 4px solid #00cc00;
        color: #006600;
    }
    </style>
""", unsafe_allow_html=True)

# Cargar modelo ONNX en caché
@st.cache_resource
def load_model():
    try:
        if not os.path.exists(MODEL_PATH):
            st.error(f" Modelo no encontrado en {MODEL_PATH}")
            return None
        
        sess = rt.InferenceSession(MODEL_PATH)
        logger.info(f"Modelo ONNX cargado exitosamente desde {MODEL_PATH}")
        return sess
    except Exception as e:
        logger.error(f"Error al cargar el modelo: {str(e)}")
        st.error(f" Error al cargar el modelo ONNX: {str(e)}")
        return None

# Función para hacer predicción
def predict_digit(image_array):
    try:
        session = load_model()
        if session is None:
            return None, None
        
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        
        result = session.run([output_name], {input_name: image_array})
        logits = result[0][0]
        
        exp_logits = np.exp(logits - np.max(logits))
        probabilities = exp_logits / np.sum(exp_logits)
        
        predicted_digit = np.argmax(probabilities)
        confidence = probabilities[predicted_digit]
        
        return predicted_digit, probabilities
    except Exception as e:
        logger.error(f"Error durante la inferencia: {str(e)}")
        st.error(f" Error durante la predicción: {str(e)}")
        return None, None

# Título principal
st.title(" MNIST - Reconocimiento de Dígitos Escritos a Mano")
st.markdown("---")

# Sidebar con información
with st.sidebar:
    st.header("Información del Modelo")
    st.markdown("""
    **Modelo:** MNIST Int8 ONNX  
    **Entrada:** Imagen 28×28 píxeles en escala de grises  
    **Salida:** 10 probabilidades (dígitos 0-9)  
    **Versión:** 1.0  
    """)
    
    st.markdown("---")
    
    # Variables de entorno
    environment = os.getenv("ENVIRONMENT", "dev")
    azure_connection = os.getenv("AZURE_STORAGE_CONNECTION_STRING", "No configurado")
    
    st.markdown(f"**Entorno:** `{environment}`")
    
    if azure_connection != "No configurado":
        st.markdown("**Azure Storage:** Conectado")
    else:
        st.markdown("**Azure Storage:** No configurado")
    
    # Opción para ver predicciones anteriores
    if st.checkbox("Ver últimas predicciones"):
        predictions = get_latest_predictions()
        if predictions:
            st.markdown("**Últimas predicciones:**")
            for pred in predictions[-10:]:
                st.caption(pred)
        else:
            st.info("No hay predicciones registradas aún")

# Contenido principal
col1, col2 = st.columns(2)

with col1:
    st.subheader("Cargar o Dibujar Imagen")
    
    # Selector de método de entrada
    input_method = st.radio(
        "Elige cómo proporcionar el dígito:",
        ["Subir imagen", "Dibujar en canvas"]
    )
    
    image_for_prediction = None
    image_display = None
    
    if input_method == "Subir imagen":
        uploaded_file = st.file_uploader(
            "Selecciona una imagen PNG, JPG o JPEG",
            type=["png", "jpg", "jpeg"]
        )
        
        if uploaded_file is not None:
            try:
                # Cargar imagen
                image = Image.open(uploaded_file)
                st.image(image, caption="Imagen original", use_column_width=True)
                
                # Procesar imagen
                image_for_prediction, image_28x28 = process_image(image)
                image_display = image_28x28
                
            except Exception as e:
                st.markdown(f"""
                <div class="error-box">
                 Error al procesar la imagen: {str(e)}
                </div>
                """, unsafe_allow_html=True)
    
    else:  # Dibujar en canvas
        st.markdown("**Dibuja un dígito en el canvas:**")
        canvas_result = st_canvas(
            fill_color="white",
            stroke_width=27,
            stroke_color="white",
            background_color="black",
            height=CANVAS_HEIGHT,
            width=CANVAS_WIDTH,
            drawing_mode="freedraw",
            key="canvas"
        )
        
        if canvas_result.image_data is not None:
            try:
                # Convertir canvas a imagen PIL
                canvas_image = Image.fromarray(canvas_result.image_data.astype('uint8'))
                
                # Procesar imagen
                image_for_prediction, image_28x28 = process_image(canvas_image, is_canvas=True)
                image_display = image_28x28
                
            except Exception as e:
                st.markdown(f"""
                <div class="error-box">
                    Error al procesar el dibujo: {str(e)}
                </div>
                """, unsafe_allow_html=True)

# Botones de acción
col_btn1, col_btn2, col_btn3 = st.columns(3)

with col_btn1:
    predict_button = st.button(" Predecir", key="predict_btn", use_container_width=True)

with col_btn2:
    clear_button = st.button(" Limpiar", key="clear_btn", use_container_width=True)

with col_btn3:
    refresh_button = st.button(" Refrescar", key="refresh_btn", use_container_width=True)

# Procesamiento de predicción
if predict_button and image_for_prediction is not None:
    with st.spinner("Prediciendo..."):
        predicted_digit, probabilities = predict_digit(image_for_prediction)
        
        if predicted_digit is not None:
            # Emoji para cada dígito
            digit_emojis = ["0️⃣", "1️⃣", "2️⃣", "3️⃣", "4️⃣", "5️⃣", "6️⃣", "7️⃣", "8️⃣", "9️⃣"]
            
            # Guardar en sesión para mostrar resultados
            st.session_state.predicted_digit = predicted_digit
            st.session_state.probabilities = probabilities
            st.session_state.image_display = image_display
            
            # Log de predicción en Azure
            timestamp = datetime.now().isoformat()
            environment = os.getenv("ENVIRONMENT", "dev")
            confidence = float(probabilities[predicted_digit])
            
            log_prediction_to_azure(
                timestamp=timestamp,
                prediction=int(predicted_digit),
                confidence=confidence,
                environment=environment
            )
            
            st.markdown("""
            <div class="success-box">
             Predicción completada exitosamente
            </div>
            """, unsafe_allow_html=True)

elif predict_button and image_for_prediction is None:
    st.markdown("""
    <div class="error-box">
     Por favor, carga o dibuja una imagen antes de predecir
    </div>
    """, unsafe_allow_html=True)

if clear_button:
    # Limpiar sesión
    st.session_state.predicted_digit = None
    st.session_state.probabilities = None
    st.session_state.image_display = None
    st.rerun()

# Mostrar resultados si existen
if "predicted_digit" in st.session_state and st.session_state.predicted_digit is not None:
    st.markdown("---")
    st.subheader(" Resultados de la Predicción")
    
    result_col1, result_col2 = st.columns(2)
    
    with result_col1:
        st.markdown("**Imagen procesada (28×28):**")
        if st.session_state.image_display is not None:
            st.image(st.session_state.image_display, width=200)
    
    with result_col2:
        st.markdown(f"""
        <div class="digit-display">
        {["0️⃣", "1️⃣", "2️⃣", "3️⃣", "4️⃣", "5️⃣", "6️⃣", "7️⃣", "8️⃣", "9️⃣"][st.session_state.predicted_digit]}
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="confidence-display">
        Confianza: {st.session_state.probabilities[st.session_state.predicted_digit]:.2%}
        </div>
        """, unsafe_allow_html=True)
    
    # Gráfico de barras con todas las probabilidades
    st.markdown("**Probabilidades por dígito:**")
    
    fig = go.Figure(data=[
        go.Bar(
            x=list(range(10)),
            y=st.session_state.probabilities,
            marker=dict(
                color=st.session_state.probabilities,
                colorscale='Viridis',
                showscale=False
            ),
            text=[f"{p:.1%}" for p in st.session_state.probabilities],
            textposition="auto",
        )
    ])
    
    fig.update_layout(
        title="Distribución de Probabilidades",
        xaxis_title="Dígito",
        yaxis_title="Probabilidad",
        height=400,
        showlegend=False,
        template="plotly_dark"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Información detallada
    st.markdown("**Top 3 predicciones:**")
    top_3_indices = np.argsort(st.session_state.probabilities)[::-1][:3]
    
    for rank, idx in enumerate(top_3_indices, 1):
        st.markdown(f"**{rank}.** Dígito `{idx}`: {st.session_state.probabilities[idx]:.2%}")

# Footer
st.markdown("---")