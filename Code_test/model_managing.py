import streamlit as st
import cv2
import numpy as np
import time
import queue
import tensorflow as tf
import os
from threading import Thread

# Disable oneDNN optimizations for reproducibility
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

class ModelManager:
    def __init__(self):
        self.process_thread = None
        self.models = {}
        self.input_queue = queue.Queue(maxsize=10)
        self.output_queue = queue.Queue()
        self.running = False
        self.last_prediction = None

    def load_model(self, model_name, model_path):
        """Load a model from local path"""
        try:
            if not os.path.exists(model_path):
                st.error(f"Model not found at: {model_path}")
                return False
                
            model = tf.keras.models.load_model(model_path)
            self.models[model_name] = model
            st.success(f"✅ Loaded: {model_name}")
            return True
        except Exception as e:
            st.error(f"❌ Failed to load {model_name}: {str(e)}")
            return False

    # ... (keep all other ModelManager methods unchanged) ...

def main():
    # Page config
    st.set_page_config(
        page_title="GuardVision Local",
        page_icon="👁️",
        layout="wide"
    )

    # Custom CSS (keep your existing styles)
    st.markdown("""
        <style>
        .main { background-color: #0E1117; }
        .stButton>button {
            width: 100%;
            border-radius: 5px;
            height: 3em;
            background-color: #FF4B4B;
            color: white;
        }
        </style>
    """, unsafe_allow_html=True)

    # Title
    st.title("👁️ GuardVision Local Mode")
    st.markdown("Real-time surveillance with local models")

    # Sidebar controls
    with st.sidebar:
        st.header("⚙️ Control Panel")
        use_action = st.checkbox("Action Recognition", value=True)
        use_crowd = st.checkbox("Crowd Density", value=True)
        conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
        stop_button = st.button("🛑 Stop System", type="primary")

    # Initialize system
    manager = ModelManager()
    
    # Load models from local paths
    if use_action:
        manager.load_model("action", "models/action_model.keras")
    if use_crowd:
        manager.load_model("crowd", "models/crowd_model.keras")

    # Start processing
    manager.start_processing()

    # Video display
    video_placeholder = st.empty()
    status_placeholder = st.empty()
    fps_placeholder = st.empty()

    cap = cv2.VideoCapture(0)
    prev_time = 0

    try:
        while cap.isOpened() and not stop_button:
            ret, frame = cap.read()
            if not ret:
                st.error("Camera error")
                break

            # FPS calculation
            current_time = time.time()
            fps = 1 / (current_time - prev_time) if prev_time > 0 else 0
            prev_time = current_time

            # Process frame
            manager.input_queue.put(frame)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Display
            video_placeholder.image(frame_rgb, 
                                  channels="RGB", 
                                  use_container_width=True)
            fps_placeholder.metric("FPS", f"{fps:.1f}")
            status_placeholder.success("🟢 Running")

    finally:
        manager.stop_processing()
        cap.release()
        status_placeholder.error("🔴 Stopped")

if __name__ == "__main__":
    main()
