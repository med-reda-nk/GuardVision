import streamlit as st
import cv2
import numpy as np
import time
import queue
import tensorflow as tf
import os
from threading import Thread
from collections import deque
import mediapipe as mp
import torch

# Disable oneDNN optimizations for reproducibility
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


class PoseDetector:
    def __init__(self, mode=False, upBody=False, smooth=True, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(
            static_image_mode=self.mode,
            smooth_landmarks=self.upBody,
            smooth_segmentation=self.smooth,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)

        if self.results.pose_landmarks and draw:
            self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img

    def findPosition(self, img, draw=True):
        lmList = []
        if self.results.pose_landmarks:
            h, w, _ = img.shape
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return lmList


class ModelManager:
    def __init__(self):
        self.process_thread = None
        self.models = {}
        self.input_queue = queue.Queue(maxsize=10)
        self.output_queue = queue.Queue()
        self.running = False
        self.last_prediction = None
        self.message_history = deque(maxlen=20)
        # Track last detections

        self.last_detections = {
            "action": None,
            "crowd": None
        }
        # Define model thresholds and class names
        self.model_config = {
            "action": {
                "threshold": 0.7,
                "class_names": ['Abuse', 'Arrest', 'Arson', 'Assault', 'Burglary', 'Explosion', 'Fighting', "Normal",
                                'Robbery', 'Shooting', 'Shoplifting', 'Stealing', 'Vandalism']
            },
            "crowd": {
                "threshold": 0.8,
                "density": 0
            }
        }
        # Initialize PoseDetector
        self.pose_detector = PoseDetector()
        self.yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

    def load_model(self, model_name, model_path):
        """Load a model from local path"""
        try:
            if not os.path.exists(model_path):
                st.error(f"Model not found at: {model_path}")
                return False

            model = tf.keras.models.load_model(model_path)
            self.models[model_name] = model
            st.success(f"{model_name} successfully loaded")
            return True
        except Exception as e:
            st.error(f"Failed to load {model_name}: {str(e)}")
            return False

    def process_frame(self, frame):
        """Process frame through models and return meaningful detections"""
        input_size = (64, 64)
        frame_resized = cv2.resize(frame, input_size)
        blob = frame_resized.astype(np.float32) / 255.0
        blob = np.expand_dims(blob, axis=0)

        detections = {}
        pose_landmarks = []  # Store all pose landmarks from detected persons

        for model_name, model in self.models.items():
            try:
                preds = model.predict(blob, verbose=0)[0]  # Get first batch item

                if model_name == "action":
                    # Get top class prediction
                    class_id = np.argmax(preds)
                    confidence = preds[class_id]
                    if confidence > self.model_config["action"]["threshold"]:
                        class_name = self.model_config["action"]["class_names"][class_id]
                        detections["action"] = {
                            "class": class_name,
                            "confidence": float(confidence)
                        }
                        self._add_detection_message(model_name, class_name, confidence)

                elif model_name == "crowd":
                    # Get density value (assuming single output)
                    density = float(preds[0][0])
                    if density > self.model_config["crowd"]["threshold"]:
                        detections["crowd"] = {
                            "density": density
                        }
                        self._add_detection_message(model_name, None, density)

            except Exception as e:
                print(f"Prediction error in {model_name}: {str(e)}")

        # YOLO detection and pose estimation
        results = self.yolo_model(frame)
        yolo_detections = results.xyxy[0]  # Get detections

        for *box, conf, cls in yolo_detections:  # Iterate through detected objects
            if conf > 0.5:  # Confidence threshold
                x1, y1, x2, y2 = map(int, box)  # Get bounding box coordinates
                cropped_img = frame[y1:y2, x1:x2]  # Crop the image

                # Detect pose in the cropped image
                cropped_img = self.pose_detector.findPose(cropped_img, draw=False)
                lmList = self.pose_detector.findPosition(cropped_img, draw=False)

                # Convert cropped coordinates back to original frame coordinates
                if lmList:
                    adjusted_landmarks = []
                    for id, cx, cy in lmList:
                        # Adjust coordinates to original frame
                        original_cx = cx + x1
                        original_cy = cy + y1
                        adjusted_landmarks.append([id, original_cx, original_cy])
                    pose_landmarks.append(adjusted_landmarks)
                    print(f'Landmarks for detected person: {len(adjusted_landmarks)} points')

        self.last_detections = detections if detections else self.last_detections

        # Return both detections and pose landmarks
        return {
            "detections": detections if detections else None,
            "pose_landmarks": pose_landmarks if pose_landmarks else None
        }

    def _add_detection_message(self, model_name, class_name, value):
        """Add message only if detection exceeds threshold and is different from last detection"""
        timestamp = time.strftime("%H:%M:%S")
        if model_name == "action":
            if self.last_detections.get("action") is None or self.last_detections["action"]["class"] != class_name:
                self.message_history.append(
                    f"🚨 [{timestamp}] Detected: {class_name} (Confidence: {value:.2f})"
                )
        elif model_name == "crowd":
            if self.last_detections.get("crowd") is None or self.last_detections["crowd"]["density"] != value:
                self.message_history.append(
                    f"👥 [{timestamp}] Crowd density: {value:.2f}"
                )

    def start_processing(self):
        self.running = True
        self.process_thread = Thread(target=self._process_loop)
        self.process_thread.start()

    def stop_processing(self):
        self.running = False
        if self.process_thread:
            self.process_thread.join()

    def _process_loop(self):
        while self.running:
            try:
                frame = self.input_queue.get(timeout=1)
                results = self.process_frame(frame)
                self.output_queue.put(results)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Processing error: {str(e)}")


def draw_predictions(frame, detections, pose_landmarks):
    """Draw detections and pose landmarks on frame if they exist"""
    if detections:
        y_offset = 30
        for model_name, detection in detections.items():
            if model_name == "action":
                text = f"Action: {detection['class']} ({detection['confidence']:.2f})"
                color = (0, 0, 255)  # Red for actions
                cv2.putText(frame, text, (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                y_offset += 30
            elif model_name == "crowd":
                text = f"Crowd: {detection['density']:.2f}"
                color = (0, 165, 255)  # Orange for crowd
                cv2.putText(frame, text, (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                y_offset += 30

    # Draw pose landmarks and connections for each detected person
    if pose_landmarks:
        for person_landmarks in pose_landmarks:
            # Convert to dictionary for easier access
            landmarks_dict = {id: (cx, cy) for id, cx, cy in person_landmarks}

            # Draw landmarks as circles
            for id, cx, cy in person_landmarks:
                cv2.circle(frame, (cx, cy), 4, (0, 255, 0), thickness=-1)

            # Draw body connections using MediaPipe pose connections
            pose_connections = [
                # Face
                (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
                # Body
                (9, 10),  # mouth
                (11, 12),  # shoulders
                (11, 13), (13, 15),  # left arm
                (12, 14), (14, 16),  # right arm
                (11, 23), (12, 24),  # torso
                (23, 24),  # hips
                # Left leg
                (23, 25), (25, 27), (27, 29), (29, 31), (27, 31),
                # Right leg
                (24, 26), (26, 28), (28, 30), (30, 32), (28, 32),
                # Hands
                (15, 17), (15, 19), (15, 21), (17, 19),
                (16, 18), (16, 20), (16, 22), (18, 20)
            ]

            # Draw connections
            for connection in pose_connections:
                start_idx, end_idx = connection
                if start_idx in landmarks_dict and end_idx in landmarks_dict:
                    start_point = landmarks_dict[start_idx]
                    end_point = landmarks_dict[end_idx]
                    cv2.line(frame, start_point, end_point, (255, 0, 0), 2)

    return frame


def display_message_history(messages):
    """Display messages only if they exist"""
    with st.expander("🔔 Detection Log", expanded=True):
        if not messages:
            st.info("System running - no alerts detected")
        else:
            for msg in reversed(messages):
                if "🚨" in msg:
                    st.error(msg)
                elif "👥" in msg:
                    st.warning(msg)


def main():
    st.set_page_config(
        page_title="GuardVision Local",
        page_icon="👁️",
        layout="wide"
    )

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

    st.title("👁️ GuardVision Local Mode")
    st.markdown("Real-time surveillance with local AI models")

    with st.sidebar:
        st.header("⚙️ Control Panel")
        use_action = st.checkbox("Action Recognition", value=True)
        use_crowd = st.checkbox("Crowd Density", value=True)
        stop_button = st.button("🛑 Stop System", type="primary")

    manager = ModelManager()

    if use_action:
        manager.load_model("action", "models/action_model.keras")
    if use_crowd:
        manager.load_model("crowd", "models/crowd_model.keras")

    manager.start_processing()

    col1, col2 = st.columns([3, 1])
    with col1:
        video_placeholder = st.empty()
        status_placeholder = st.empty()
    with col2:
        st.subheader("Detection Log")
        message_placeholder = st.empty()

    cap = cv2.VideoCapture(0)
    prev_time = 0

    try:
        while cap.isOpened() and not stop_button:
            ret, frame = cap.read()
            if not ret:
                st.error("Camera error")
                break

            current_time = time.time()
            fps = 1 / (current_time - prev_time) if prev_time > 0 else 0
            prev_time = current_time

            manager.input_queue.put(frame)

            results = None
            if not manager.output_queue.empty():
                results = manager.output_queue.get()
                manager.last_prediction = results

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if results:
                detections = results.get("detections")
                pose_landmarks = results.get("pose_landmarks")
                frame_rgb = draw_predictions(frame_rgb, detections, pose_landmarks)

            video_placeholder.image(frame_rgb, channels="RGB", width=None)
            status_placeholder.success(f"🟢 Running | FPS: {fps:.1f}")
            display_message_history(manager.message_history)

    finally:
        manager.stop_processing()
        cap.release()
        status_placeholder.error("🔴 Stopped")


if __name__ == "__main__":
    main()
