import streamlit as st
import numpy as np
import requests
import matplotlib.pyplot as plt
from io import BytesIO

# Set page config
st.set_page_config(
    page_title="Radar Target Classification",
    page_icon="🎯",
    layout="wide"
)

def main():
    st.title("Radar Target Classification")
    st.write("Upload radar signal data for classification")
    
    # File upload
    uploaded_file = st.file_uploader("Choose a file", type=['npy', 'bin'])
    
    if uploaded_file is not None:
        # Read file
        signal_data = np.load(uploaded_file) if uploaded_file.type == 'application/x-npy' \
                     else np.frombuffer(uploaded_file.read(), dtype=np.float32)
        
        # Display signal plot
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(signal_data)
        ax.set_title("Radar Signal")
        ax.set_xlabel("Time")
        ax.set_ylabel("Amplitude")
        st.pyplot(fig)
        
        # Make prediction
        if st.button("Classify"):
            # Convert signal to bytes
            signal_bytes = BytesIO()
            np.save(signal_bytes, signal_data)
            
            # Send to API
            files = {'file': ('signal.npy', signal_bytes.getvalue())}
            response = requests.post('http://localhost:8000/predict', files=files)
            
            if response.status_code == 200:
                result = response.json()
                
                # Display results
                st.success(f"Predicted Class: {result['class']}")
                st.info(f"Confidence: {result['confidence']:.2%}")
            else:
                st.error("Error making prediction")

if __name__ == "__main__":
    main() 