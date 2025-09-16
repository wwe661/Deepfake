import requests
import streamlit as st
import time
from PIL import Image
import io
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Read threshold (default to 0.5 if not set)
THRESHOLD = float(os.getenv("THRESHOLD", 0.5))

# Configure page
st.set_page_config(
    page_title="Is It Real? - Authentication Check",
    page_icon="🔍",
    layout="centered"
)

# Custom styles
def load_css():
    st.markdown("""
        <style>
            .main {
                max-width: 800px;
                padding: 2rem;
            }
            .stButton>button {
                background: linear-gradient(90deg, #4b6cb7 0%, #182848 100%);
                color: white;
                border: none;
                padding: 0.5rem 1.5rem;
                border-radius: 8px;
                font-weight: 600;
            }
            .stButton>button:hover {
                background: linear-gradient(90deg, #3a56a5 0%, #121f3d 100%);
                color: white;
            }
            .file-uploader {
                border: 2px dashed #4b6cb7;
                border-radius: 8px;
                padding: 2rem;
                text-align: center;
                margin: 1rem 0;
            }
            .result-box {
                padding: 1.5rem;
                border-radius: 8px;
                margin-top: 1.5rem;
                margin-bottom: 1.5rem;
                border: 1px solid #e0e0e0;
                text-align: center;
            }
            .real {
                # background-color: #e6f7e6;
                border: 1px solid #4CAF50;
            }
            .fake {
                # background-color: #ffebee;
                border: 1px solid #f44336;
            }
            .title {
                text-align: center;
                margin-bottom: 1.5rem;
                color: #182848;
            }
        </style>
    """, unsafe_allow_html=True)

load_css()

# App content
st.markdown('<h1 class="title">🔍 Is It Real?</h1>', unsafe_allow_html=True)
st.markdown("Upload an image, or video to check its authenticity")

# File upload section
uploaded_file = st.file_uploader(
    "", 
    type=["jpg", "jpeg", "png", "gif", "mp4",],
    accept_multiple_files=False,
    help="Upload a file to check its authenticity"
)

# Result state
result = None
confidence = 0

# Check button and processing
if st.button("Check Authenticity"):
    if uploaded_file is not None:
        
        # Show loading state
        with st.spinner("Analyzing file contents..."):
            # Simulate processing time
            time.sleep(2)
            
            # Send file to backend for prediction
            print("predicting...")
            files = {"file": uploaded_file.getvalue()}
            # _ = requests.post("http://localhost:5000/debug_preprocess", files={"file": uploaded_file})  # for debugging
            response = requests.post("http://localhost:5000/predict", files={"file": uploaded_file})

            if response.status_code == 200:
                result = response.json()["prediction"][0]
                print("response :",response.json())
                result = round(result[0],4)
                debug_images = response.json().get("paths", {})
                st.success(f"Prediction: {result}")
            else:
                st.error("Error: Could not get prediction")
                
            print(result)
            
            # Random result for demo (in real app, replace with your detection logic)
            is_real = True if  result > THRESHOLD else False
            confidence = result * 100 if is_real else (1 - result) * 100
            result = {
                "is_real": is_real,
                "confidence": confidence,
                "file_type": uploaded_file.type,
                "file_size": f"{len(uploaded_file.getvalue()) / 1024:.2f} KB"
            }
    else:
        st.warning("Please upload a file first")

# Display results
if result:
    st.markdown("### Analysis Results")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("File Type", result["file_type"].split("/")[-1].upper())
    with col2:
        st.metric("File Size", result["file_size"])
    
    result_class = "real" if result["is_real"] else "fake"
    result_text = "✅ Authentic" if result["is_real"] else "❌ Potential Fake"
    
    st.markdown(
        f'<div class="result-box {result_class}">'
        f'<h3>{result_text}</h3>'
        f'<p>Confidence: {result["confidence"]}%</p>'
        '</div>', 
        unsafe_allow_html=True
    )
    
    # Display uploaded image if it's an image file
    if "image" in result["file_type"]:
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
    
    # Add detailed analysis section (placeholder for real implementation)
    # st.expander("Detailed Analysis").write("""
    #     In a real implementation, this section would contain:
    #     - EXIF metadata analysis for images
    #     - Error level analysis for potential manipulation
    #     - Source verification for documents
    #     - Hash comparison with known databases
    #     - Digital signature verification
    # """)
    with st.expander("Detailed Analysis"):
        st.write("Preprocessing steps visualized:")
        name_list = ["original", "clahe", "sharpened", "resized", "Color_converted", "final"]
        for name in name_list:
            url = debug_images.get(name)
            if url:
                st.image(url, caption=name, use_container_width=False)

# Empty state instructions
elif uploaded_file is None:
    st.markdown("""
    <div style="text-align: center; margin-top: 2rem; color: #666;">
        <p>This tool helps detect manipulated images or documents.</p>
        <p>Upload a file to begin analysis.</p>
    </div>
    """, unsafe_allow_html=True)
