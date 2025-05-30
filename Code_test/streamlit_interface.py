    import streamlit as st
    import cv2
    import numpy as np
    from PIL import Image
    import time
    import queue

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
        st.title("🤖 AI Vision System")
        st.markdown("""
            Real-time object detection and face recognition system powered by YOLOv3 and custom face detection models.
        """)

        # Sidebar controls
        with st.sidebar:
            st.header("⚙️ Controls")
            
            # Model selection
            st.subheader("Model Selection")
            use_action = st.checkbox("Action Recognation", value=True)
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
            stop_button = st.button("🛑 Stop Processing", type="primary")

        # Initialize model manager
        manager = ModelManager()
        
        # Load models
        if use_yolo:
            manager.load_model("action", "action_model.keras")
        if use_face:
            manager.load_model("crowd", "crowd_model.keras")
        
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
            fps = 1/(current_time - prev_time) if prev_time > 0 else 0
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

