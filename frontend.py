import streamlit as st
import requests
from PIL import Image
import io

# Page Configuration
st.set_page_config(
    page_title="Kidney Disease Classifier",
    page_icon="🏥",
    layout="centered"
)

# Application Title
st.title("🏥 Kidney Disease Classification Portal")
st.write("Upload a CT scan image to predict the diagnostic condition (Cyst, Normal, Tumor, or Stone).")

st.markdown("---")

# File uploader on Streamlit
uploaded_file = st.file_uploader("Choose a CT Scan image (JPG/PNG)...", type=["jpg", "jpeg", "png"])

# FastAPI endpoint address
API_URL = "http://127.0.0.1:8000/predict"

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded CT Scan Image", use_column_width=True)
    
    st.write("")
    
    # Prediction trigger button
    if st.button("Generate Diagnostic Prediction", type="primary"):
        with st.spinner("Analyzing image features..."):
            try:
                # Convert the PIL image back to bytes to send via requests API
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format=image.format if image.format else "JPEG")
                img_byte_arr = img_byte_arr.getvalue()
                
                # Format payload
                files = {"file": (uploaded_file.name, img_byte_arr, uploaded_file.type)}
                
                # Send request to FastAPI
                response = requests.post(API_URL, files=files)
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Highlighted result
                    st.success(f"**Prediction:** {result['prediction']}")
                    st.info(f"**Confidence Level:** {result['confidence']}")
                    
                    # Probability break down chart
                    st.write("### Class Probability Breakdown")
                    st.bar_chart(result['probabilities'])
                else:
                    st.error(f"Error from server: {response.json().get('error', 'Unknown Error')}")
                    
            except requests.exceptions.ConnectionError:
                st.error("Could not connect to backend FastAPI. Is the server running on port 8000?")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")