import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time
import queue
import tensorflow as tf
import os
from threading import Thread

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

class ModelManager:
    def __init__(self):
        self.process_thread = None
        self.models = {}
        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()
        self.running = False

    def load_model(self, model_name, model_path):
        """Load a model from the specified path"""
        try:
            model = tf.keras.models.load_model(model_path)
            self.models[model_name] = model
            print(f"Successfully loaded model: {model_name}")
        except Exception as e:
            print(f"Error loading model {model_name}: {str(e)}")

    def process_frame(self, frame):
        """Process a single frame through all loaded Keras models and combine predictions"""
        # Prepare input blob
        input_size = (64, 64)  # Change this to your model's input size
        frame_resized = cv2.resize(frame, input_size)

        # Normalize the image (if required by your model)
        blob = frame_resized.astype(np.float32) / 255.0  # Normalize to [0, 1]

        # Expand dimensions to match the model input shape (e.g., (1, height, width, channels))
        blob = np.expand_dims(blob, axis=0)

        # Collect predictions from all models
        predictions = []
        for model_name, model in self.models.items():
            try:
                preds = model.predict(blob)
                predictions.append(preds)
            except Exception as e:
                print(f"Error predicting with model {model_name}: {str(e)}")

        # Combine predictions (e.g., average them)
        if predictions:
            combined_predictions = np.mean(predictions, axis=0)  # Average predictions
            return combined_predictions
        else:
            return None

    def start_processing(self):
        """Start the processing thread"""
        self.running = True
        self.process_thread = Thread(target=self._process_loop)
        self.process_thread.start()

    def stop_processing(self):
        """Stop the processing thread"""
        self.running = False
        if hasattr(self, 'process_thread'):
            self.process_thread.join()

    def _process_loop(self):
        """Main processing loop"""
        while self.running:
            try:
                frame = self.input_queue.get(timeout=1)
                results = self.process_frame(frame)
                self.output_queue.put(results)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in processing loop: {str(e)}")


def main():
    # Set page config
    st.set_page_config(
        page_title="GuardVision",
        page_icon="👁️",
        layout="wide"
    )

    # Custom CSS
    st.markdown("""
            <style>
            .main {
                background-color: #f5f5f5;
            }
            .stButton>button {
                width: 100%;
                border-radius: 5px;
                height: 3em;
            }
            .css-1d391kg {
                padding: 1rem;
            }
            </style>
        """, unsafe_allow_html=True)

    # Title and description
    st.title("GuardViision")
    st.markdown("""
            Real-time Surveillance System.
        """)

    # Sidebar controls
    with st.sidebar:
        st.header("Controls Panel")

        # Model selection
        st.subheader("Model Selection")
        use_action = st.checkbox("Action Recognition", value=True)
        use_crowd = st.checkbox("Crowd Density", value=True)

        # Confidence threshold
        st.subheader("Detection Settings")
        conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)

        # Display options
        st.subheader("Display Options")
        show_fps = st.checkbox("Show FPS", value=True)
        show_boxes = st.checkbox("Show Detection Boxes", value=True)

        # Status indicators
        st.subheader("System Status")
        status_placeholder = st.empty()

        # Stop button
        stop_button = st.button("Stop Processing", type="primary")

    # Initialize model manager
    manager = ModelManager()

    # Load models
    if use_action:
        manager.load_model("action", "models/action_model.keras")
    if use_crowd:
        manager.load_model("crowd", "models/crowd_model.keras")

    # Start processing
    manager.start_processing()

    # Create columns for video feed and stats
    col1, col2 = st.columns([3, 1])

    with col1:
        st.subheader("📹 Live Feed")
        stframe = st.empty()

    with col2:
        st.subheader("📊 Statistics")
        fps_placeholder = st.empty()
        detections_placeholder = st.empty()

    # Initialize video capture
    cap = cv2.VideoCapture(0)

    # FPS calculation variables
    prev_time = 0
    fps = 0

    while cap.isOpened() and not stop_button:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to access camera")
            break

        # Calculate FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time) if prev_time > 0 else 0
        prev_time = current_time

        # Add frame to processing queue
        manager.input_queue.put(frame)

        # Get results if available
        try:
            results = manager.output_queue.get_nowait()
            # Update detection statistics
            detections_placeholder.metric("Detections", len(results))
        except queue.Empty:
            pass

        # Convert frame to RGB for display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Display FPS if enabled
        if show_fps:
            cv2.putText(frame_rgb, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the frame
        stframe.image(frame_rgb, channels="RGB", use_column_width=True)

        # Update status
        status_placeholder.success("System Running")

        # Update FPS display
        fps_placeholder.metric("FPS", f"{fps:.1f}")

        # Add a small delay to prevent overwhelming the system
        time.sleep(0.01)

    # Cleanup
    manager.stop_processing()
    cap.release()
    status_placeholder.error("System Stopped")


if __name__ == "__main__":
    main()

# To run the Streamlit app, use the following command in your terminal:
#streamlit run app.py --server.port 8501 --server.address 0.0.0.0
