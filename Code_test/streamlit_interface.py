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
from datetime import datetime, timedelta
import json
from collections import Counter, defaultdict
import re

# Disable oneDNN optimizations for reproducibility
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"



# Add this class after OptimizedPoseDetector class
class NLPReportGenerator:
    def __init__(self):
        self.daily_events = defaultdict(list)
        self.session_start_time = datetime.now()
        self.threat_keywords = {
            'high_threat': ['shooting', 'assault', 'fighting', 'abuse', 'explosion', 'arson'],
            'medium_threat': ['robbery', 'burglary', 'stealing', 'shoplifting', 'vandalism'],
            'crowd_related': ['crowd', 'density', 'gathering']
        }

    def log_event(self, event_type, details, timestamp=None):
        """Log events for daily report generation"""
        if timestamp is None:
            timestamp = datetime.now()

        event_data = {
            'timestamp': timestamp,
            'type': event_type,
            'details': details,
            'hour': timestamp.hour
        }

        date_key = timestamp.strftime('%Y-%m-%d')
        self.daily_events[date_key].append(event_data)

    def generate_daily_summary(self, target_date=None):
        """Generate NLP-based daily summary report"""
        if target_date is None:
            target_date = datetime.now().strftime('%Y-%m-%d')

        events = self.daily_events.get(target_date, [])

        if not events:
            return "No significant events recorded for today."

        # Analyze events
        analysis = self._analyze_events(events)

        # Generate natural language report
        report = self._generate_nlp_report(analysis, target_date)

        return report

    def _analyze_events(self, events):
        """Analyze events and extract patterns"""
        analysis = {
            'total_events': len(events),
            'threat_levels': {'high': 0, 'medium': 0, 'low': 0},
            'hourly_distribution': defaultdict(int),
            'event_types': Counter(),
            'peak_hours': [],
            'threat_summary': [],
            'crowd_events': 0,
            'duration_minutes': 0
        }

        # Calculate session duration
        if events:
            start_time = min(event['timestamp'] for event in events)
            end_time = max(event['timestamp'] for event in events)
            analysis['duration_minutes'] = int((end_time - start_time).total_seconds() / 60)

        for event in events:
            # Categorize threat level
            event_detail = event['details'].lower()

            if any(keyword in event_detail for keyword in self.threat_keywords['high_threat']):
                analysis['threat_levels']['high'] += 1
                analysis['threat_summary'].append(
                    f"High threat: {event['details']} at {event['timestamp'].strftime('%H:%M')}")
            elif any(keyword in event_detail for keyword in self.threat_keywords['medium_threat']):
                analysis['threat_levels']['medium'] += 1
                analysis['threat_summary'].append(
                    f"Medium threat: {event['details']} at {event['timestamp'].strftime('%H:%M')}")
            else:
                analysis['threat_levels']['low'] += 1

            # Track crowd events
            if any(keyword in event_detail for keyword in self.threat_keywords['crowd_related']):
                analysis['crowd_events'] += 1

            # Hourly distribution
            analysis['hourly_distribution'][event['hour']] += 1

            # Event types
            analysis['event_types'][event['type']] += 1

        # Find peak hours
        if analysis['hourly_distribution']:
            max_events = max(analysis['hourly_distribution'].values())
            analysis['peak_hours'] = [hour for hour, count in analysis['hourly_distribution'].items()
                                      if count == max_events]

        return analysis

    def _generate_nlp_report(self, analysis, date):
        """Generate natural language report using NLP techniques"""
        report_parts = []

        # Header
        report_parts.append(f"📊 DAILY SURVEILLANCE REPORT - {date}")
        report_parts.append("=" * 50)

        # Executive Summary
        total_events = analysis['total_events']
        duration = analysis['duration_minutes']

        if total_events == 0:
            return "No security events detected during the monitoring period."

        # Threat assessment
        high_threats = analysis['threat_levels']['high']
        medium_threats = analysis['threat_levels']['medium']

        if high_threats > 0:
            threat_status = "🔴 HIGH RISK"
            summary = f"Critical security situation detected with {high_threats} high-priority threat(s)."
        elif medium_threats > 0:
            threat_status = "🟡 MEDIUM RISK"
            summary = f"Moderate security concerns identified with {medium_threats} medium-priority event(s)."
        else:
            threat_status = "🟢 LOW RISK"
            summary = "Routine monitoring period with no significant security threats."

        report_parts.append(f"\n🎯 THREAT ASSESSMENT: {threat_status}")
        report_parts.append(f"📝 SUMMARY: {summary}")

        # Operational metrics
        report_parts.append(f"\n📈 OPERATIONAL METRICS:")
        report_parts.append(f"   • Monitoring Duration: {duration} minutes")
        report_parts.append(f"   • Total Events Detected: {total_events}")
        report_parts.append(f"   • Events per Hour: {total_events / max(1, duration / 60):.1f}")

        # Activity patterns
        if analysis['peak_hours']:
            peak_times = [f"{hour:02d}:00" for hour in analysis['peak_hours']]
            report_parts.append(f"   • Peak Activity Hours: {', '.join(peak_times)}")

        # Threat breakdown
        if high_threats > 0 or medium_threats > 0:
            report_parts.append(f"\n⚠️  THREAT BREAKDOWN:")
            if high_threats > 0:
                report_parts.append(f"   • High Priority Threats: {high_threats}")
            if medium_threats > 0:
                report_parts.append(f"   • Medium Priority Threats: {medium_threats}")

        # Crowd analysis
        if analysis['crowd_events'] > 0:
            crowd_percentage = (analysis['crowd_events'] / total_events) * 100
            report_parts.append(f"\n👥 CROWD ANALYSIS:")
            report_parts.append(f"   • Crowd-related Events: {analysis['crowd_events']} ({crowd_percentage:.1f}%)")

        # Detailed threat log
        if analysis['threat_summary']:
            report_parts.append(f"\n🚨 CRITICAL EVENTS LOG:")
            for threat in analysis['threat_summary'][:5]:  # Show top 5
                report_parts.append(f"   • {threat}")
            if len(analysis['threat_summary']) > 5:
                report_parts.append(f"   • ... and {len(analysis['threat_summary']) - 5} more events")

        # Recommendations
        recommendations = self._generate_recommendations(analysis)
        if recommendations:
            report_parts.append(f"\n💡 RECOMMENDATIONS:")
            for rec in recommendations:
                report_parts.append(f"   • {rec}")

        return "\n".join(report_parts)

    def _generate_recommendations(self, analysis):
        """Generate contextual recommendations based on analysis"""
        recommendations = []

        high_threats = analysis['threat_levels']['high']
        medium_threats = analysis['threat_levels']['medium']
        crowd_events = analysis['crowd_events']
        peak_hours = analysis['peak_hours']

        if high_threats > 0:
            recommendations.append("Immediate security response required for high-priority threats")
            recommendations.append("Consider increasing security personnel during peak hours")

        if medium_threats > 5:
            recommendations.append("Review security protocols for theft prevention")

        if crowd_events > analysis['total_events'] * 0.3:
            recommendations.append("Implement crowd management strategies")
            recommendations.append("Monitor for overcrowding during peak periods")

        if peak_hours:
            peak_str = ', '.join([f"{hour:02d}:00" for hour in peak_hours])
            recommendations.append(f"Increase surveillance focus during peak hours: {peak_str}")

        if not recommendations:
            recommendations.append("Continue routine monitoring protocols")

        return recommendations






# Add this function to display daily report in sidebar
def display_daily_report_section(manager):
    """Display daily report section in sidebar"""
    if manager and hasattr(manager, 'nlp_reporter'):
        st.subheader("📋 Daily Report")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("📊 Generate Report", key="generate_report_button"):
                report = manager.nlp_reporter.generate_daily_summary()
                st.session_state.daily_report = report

        with col2:
            if st.button("💾 Save Report", key="save_report_button"):
                if hasattr(st.session_state, 'daily_report'):
                    filename = f"surveillance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                    st.download_button(
                        label="Download Report",
                        data=st.session_state.daily_report,
                        file_name=filename,
                        mime="text/plain"
                    )

        # Display report if generated
        if hasattr(st.session_state, 'daily_report'):
            with st.expander("📄 View Report", expanded=False):
                st.text(st.session_state.daily_report)


class OptimizedPoseDetector:
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
            model_complexity=1,
            smooth_landmarks=self.smooth,
            enable_segmentation=False,
            smooth_segmentation=False,
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
                    cv2.circle(img, (cx, cy), 3, (255, 0, 0), cv2.FILLED)
        return lmList


class OptimizedModelManager:
    def __init__(self):
        self.process_thread = None
        self.models = {}
        self.tflite_models = {}
        self.input_queue = queue.Queue(maxsize=3)
        self.output_queue = queue.Queue(maxsize=3)
        self.running = False
        self.last_prediction = None
        self.message_history = deque(maxlen=20)
        self.frame_skip_counter = 0
        self.process_every_n_frames = 2

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
                "threshold": 0.5,  # Reasonable threshold for crowd detection
                "density": 0
            }
        }

        # Initialize optimized components
        self.pose_detector = OptimizedPoseDetector()

        # Load YOLOv5 with optimizations
        try:
            self.yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5n')
            self.yolo_model.conf = 0.6
            self.yolo_model.iou = 0.45
            self.yolo_model.classes = [0]  # Only detect persons

            # Set device
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.yolo_model.to(self.device)
            print(f"Using device: {self.device}")
        except Exception as e:
            print(f"Error loading YOLO model: {str(e)}")
            self.yolo_model = None

    def load_model(self, model_name, model_path):
        """Load and optimize model with TensorFlow Lite"""
        try:
            if not os.path.exists(model_path):
                st.error(f"Model not found at: {model_path}")
                return False

            # Load original Keras model
            model = tf.keras.models.load_model(model_path)

            # DEBUG: Print model information
            print(f"\n=== DEBUG: {model_name} Model Info ===")
            print(f"Input shape: {model.input_shape}")
            print(f"Output shape: {model.output_shape}")
            model.summary()
            print("=" * 50)

            # Convert to TensorFlow Lite for optimization
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]

            # Convert to TFLite
            tflite_model = converter.convert()

            # Create interpreter
            interpreter = tf.lite.Interpreter(model_content=tflite_model)
            interpreter.allocate_tensors()

            # Get input and output details
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()

            # DEBUG: Print TFLite details
            print(f"\n=== DEBUG: {model_name} TFLite Details ===")
            print(f"Input details: {input_details}")
            print(f"Output details: {output_details}")
            print("=" * 50)

            # Store TFLite model
            self.tflite_models[model_name] = {
                'interpreter': interpreter,
                'input_details': input_details,
                'output_details': output_details
            }

            st.success(f"{model_name} successfully loaded")
            return True

        except Exception as e:
            st.error(f"Failed to load {model_name}: {str(e)}")
            print(f"Full error: {e}")
            return False

    def process_frame(self, frame):
        """Process frame through models with enhanced debugging"""
        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (320, 240))

        detections = {}
        pose_landmarks = []

        # Process with TensorFlow Lite models - BOTH models will be processed
        for model_name, model_info in self.tflite_models.items():
            try:
                interpreter = model_info['interpreter']
                input_details = model_info['input_details']
                output_details = model_info['output_details']

                # Get expected input shape
                input_shape = input_details[0]['shape']
                expected_height, expected_width = input_shape[1], input_shape[2]

                # Resize frame to expected input size
                frame_resized = cv2.resize(frame, (expected_width, expected_height))
                blob = frame_resized.astype(np.float32) / 255.0
                blob = np.expand_dims(blob, axis=0)

                # Set input tensor
                interpreter.set_tensor(input_details[0]['index'], blob)

                # Run inference
                interpreter.invoke()

                # Get output
                preds = interpreter.get_tensor(output_details[0]['index'])

                # Process ACTION model predictions
                if model_name == "action":
                    print(f"\n=== DEBUG: Action Model Output ===")
                    print(f"Raw prediction shape: {preds.shape}")
                    print(f"Raw prediction values: {preds}")

                    # Get top class prediction
                    if len(preds.shape) > 1:
                        preds = preds[0]  # Remove batch dimension if present

                    class_id = np.argmax(preds)
                    confidence = preds[class_id]

                    print(f"Predicted class ID: {class_id}")
                    print(f"Confidence: {confidence}")
                    print(f"Threshold: {self.model_config['action']['threshold']}")
                    print(f"Above threshold: {confidence > self.model_config['action']['threshold']}")

                    if confidence > self.model_config["action"]["threshold"]:
                        class_name = self.model_config["action"]["class_names"][class_id]
                        detections["action"] = {
                            "class": class_name,
                            "confidence": float(confidence)
                        }
                        self._add_detection_message(model_name, class_name, confidence)
                        print(f"ACTION DETECTED: {class_name} ({confidence:.3f})")
                    else:
                        # Still show what was detected even if below threshold
                        class_name = self.model_config["action"]["class_names"][class_id]
                        print(f"Action below threshold: {class_name} ({confidence:.3f})")
                    print("=" * 50)

                # Process CROWD model predictions
                elif model_name == "crowd":
                    print(f"\n=== DEBUG: Crowd Model Output ===")
                    print(f"Raw prediction shape: {preds.shape}")
                    print(f"Raw prediction values: {preds}")

                    # Handle different output shapes
                    if len(preds.shape) == 4:  # Density map (batch, height, width, channels)
                        density_map = preds[0]
                        if len(density_map.shape) == 3:
                            density_map = density_map[:, :, 0]  # Take first channel

                        # Calculate total density and max density
                        total_density = np.sum(density_map)
                        max_density = np.max(density_map)
                        mean_density = np.mean(density_map)

                        print(f"Density map shape: {density_map.shape}")
                        print(f"Total density: {total_density}")
                        print(f"Max density: {max_density}")
                        print(f"Mean density: {mean_density}")

                        # Use total density as the crowd measure
                        density = float(total_density)

                    elif len(preds.shape) == 2:  # Single value output (batch, 1)
                        density = float(preds[0, 0])
                        print(f"Single density value: {density}")

                    elif len(preds.shape) == 1:  # Single value output (1,)
                        density = float(preds[0])
                        print(f"Single density value: {density}")

                    else:
                        print(f"Unexpected output shape: {preds.shape}")
                        density = float(np.sum(preds))
                        print(f"Sum of all values: {density}")

                    print(f"Threshold: {self.model_config['crowd']['threshold']}")
                    print(f"Density > Threshold: {density > self.model_config['crowd']['threshold']}")

                    if density > self.model_config["crowd"]["threshold"]:
                        detections["crowd"] = {
                            "density": density
                        }
                        self._add_detection_message(model_name, None, density)
                        print(f"CROWD DETECTED: Density {density:.4f}")
                    else:
                        # DEBUG: Still log low density values
                        print(f"Crowd density below threshold: {density:.4f}")
                    print("=" * 50)

            except Exception as e:
                print(f"TFLite prediction error in {model_name}: {str(e)}")
                import traceback
                traceback.print_exc()

        # YOLO detection (same as before)
        if self.yolo_model is not None:
            try:
                with torch.no_grad():
                    results = self.yolo_model(small_frame, size=320)
                    yolo_detections = results.xyxy[0].cpu().numpy()

                scale_x = frame.shape[1] / small_frame.shape[1]
                scale_y = frame.shape[0] / small_frame.shape[0]

                max_detections = 5
                processed_count = 0

                for detection in yolo_detections:
                    if processed_count >= max_detections:
                        break

                    x1, y1, x2, y2, conf, cls = detection

                    if conf > 0.5 and cls == 0:
                        x1, y1, x2, y2 = int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y)
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

                        if x2 > x1 and y2 > y1:
                            cropped_img = frame[y1:y2, x1:x2]

                            if cropped_img.shape[0] > 50 and cropped_img.shape[1] > 50:
                                cropped_img = self.pose_detector.findPose(cropped_img, draw=False)
                                lmList = self.pose_detector.findPosition(cropped_img, draw=False)

                                if lmList:
                                    adjusted_landmarks = []
                                    for id, cx, cy in lmList:
                                        original_cx = cx + x1
                                        original_cy = cy + y1
                                        adjusted_landmarks.append([id, original_cx, original_cy])
                                    pose_landmarks.append(adjusted_landmarks)

                        processed_count += 1

            except Exception as e:
                print(f"YOLO processing error: {str(e)}")

        # Update last detections
        if detections:
            self.last_detections.update(detections)

        return {
            "detections": detections if detections else None,
            "pose_landmarks": pose_landmarks if pose_landmarks else None
        }

    def initialize_nlp_reporter(self):
        """Initialize NLP report generator"""
        self.nlp_reporter = NLPReportGenerator()

    def _add_detection_message(self, model_name, class_name, value):
        """Add message and log event for NLP reporting"""
        timestamp = time.strftime("%H:%M:%S")
        current_time = datetime.now()

        if model_name == "action":
            if (self.last_detections.get("action") is None or
                    self.last_detections["action"].get("class") != class_name):
                message = f"🚨 [{timestamp}] Action Detected: {class_name} (Confidence: {value:.2f})"
                self.message_history.append(message)

                # Log event for NLP report
                if hasattr(self, 'nlp_reporter'):
                    self.nlp_reporter.log_event(
                        event_type="action_detection",
                        details=f"{class_name} detected with {value:.2f} confidence",
                        timestamp=current_time
                    )

        elif model_name == "crowd":
            if (self.last_detections.get("crowd") is None or
                    abs(self.last_detections["crowd"].get("density", 0) - value) > 0.1):
                message = f"👥 [{timestamp}] Crowd density: {value:.4f}"
                self.message_history.append(message)

                # Log event for NLP report
                if hasattr(self, 'nlp_reporter'):
                    self.nlp_reporter.log_event(
                        event_type="crowd_detection",
                        details=f"Crowd density level {value:.4f}",
                        timestamp=current_time
                    )

    def start_processing(self):
        self.running = True
        self.process_thread = Thread(target=self._process_loop, daemon=True)
        self.process_thread.start()

    def stop_processing(self):
        self.running = False
        if self.process_thread:
            self.process_thread.join(timeout=2)

    def _process_loop(self):
        while self.running:
            try:
                frame = self.input_queue.get(timeout=0.5)
                results = self.process_frame(frame)

                try:
                    self.output_queue.put_nowait(results)
                except queue.Full:
                    pass

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Processing error: {str(e)}")


def draw_predictions(frame, detections, pose_landmarks):
    """Draw only pose landmarks on frame - model predictions shown in sidebar"""
    # Only draw pose landmarks on the frame
    if pose_landmarks:
        for person_landmarks in pose_landmarks:
            key_points = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
            landmarks_dict = {id: (cx, cy) for id, cx, cy in person_landmarks}

            for id, cx, cy in person_landmarks:
                if id in key_points:
                    cv2.circle(frame, (cx, cy), 3, (0, 255, 0), thickness=-1)

            connections = [
                (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
                (11, 23), (12, 24), (23, 24), (23, 25), (25, 27),
                (24, 26), (26, 28),
            ]

            for start_idx, end_idx in connections:
                if start_idx in landmarks_dict and end_idx in landmarks_dict:
                    start_point = landmarks_dict[start_idx]
                    end_point = landmarks_dict[end_idx]
                    cv2.line(frame, start_point, end_point, (255, 0, 0), 2)

    return frame


def display_message_history(messages):
    """Display messages efficiently"""
    if not messages:
        st.info("System running - no alerts detected")
    else:
        recent_messages = list(messages)[-10:]
        for msg in reversed(recent_messages):
            if "🚨" in msg:
                st.error(msg)
            elif "👥" in msg:
                st.warning(msg)


def main():
    st.set_page_config(
        page_title="GuardVision - Dual Model System",
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

    st.title("👁️ GuardVision")
    st.markdown("**Surveillance System**")

    # Initialize session state
    if 'manager' not in st.session_state:
        st.session_state.manager = None
        st.session_state.cap = None
        st.session_state.running = False

    with st.sidebar:
        st.header("⚙️ Control Panel")

        # Model selection - BOTH enabled by default
        use_action = st.checkbox("Action Recognition", value=True)
        use_crowd = st.checkbox("Crowd Density", value=True)

        # Model thresholds
        st.subheader("Model Thresholds")
        action_threshold = st.slider("Action Confidence Threshold", 0.1, 1.0, 0.7, 0.05)
        crowd_threshold = st.slider("Crowd Density Threshold", 0.1, 2.0, 0.5, 0.1)

        # Performance settings
        st.subheader("Performance Settings")
        frame_skip = st.slider("Frame Skip", 1, 10, 3)

        # Camera settings
        st.subheader("Camera Settings")
        camera_resolution = st.selectbox("Resolution", ["320x240", "640x480", "1280x720"], index=1)

        # Model status
        st.subheader("📊 Model Status")
        display_daily_report_section(st.session_state.manager)

        if st.session_state.manager and st.session_state.manager.tflite_models:
            # Daily Report Section
            for model_name in st.session_state.manager.tflite_models.keys():
                st.success(f"✅ {model_name.capitalize()} Model Loaded")

        # Control buttons
        col1, col2 = st.columns(2)
        with col1:
            start_button = st.button("🚀 Start", type="primary")
        with col2:
            stop_button = st.button("🛑 Stop", type="secondary")

    # Handle start button
    if start_button and not st.session_state.running:
        if not use_action and not use_crowd:
            st.error("Please select at least one model to use!")
        else:
            st.session_state.manager = OptimizedModelManager()
            st.session_state.manager.initialize_nlp_reporter()

            # Update thresholds
            st.session_state.manager.model_config["action"]["threshold"] = action_threshold
            st.session_state.manager.model_config["crowd"]["threshold"] = crowd_threshold

            # Load models
            models_loaded = False
            models_to_load = []

            if use_action:
                models_to_load.append(("action", "models/action_model.keras"))
            if use_crowd:
                models_to_load.append(("crowd", "models/crowd_model.keras"))

            with st.spinner("Loading models..."):
                for model_name, model_path in models_to_load:
                    if st.session_state.manager.load_model(model_name, model_path):
                        models_loaded = True
                    else:
                        st.error(f"Failed to load {model_name} model from {model_path}")

            if models_loaded:
                st.session_state.manager.start_processing()

                try:
                    st.session_state.cap = cv2.VideoCapture(0)

                    if not st.session_state.cap.isOpened():
                        st.error("Could not open camera. Please check if camera is available.")
                        st.session_state.running = False
                    else:
                        width, height = map(int, camera_resolution.split('x'))
                        st.session_state.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                        st.session_state.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                        st.session_state.cap.set(cv2.CAP_PROP_FPS, 30)

                        st.session_state.running = True

                        # Show which models are running
                        active_models = [name for name in st.session_state.manager.tflite_models.keys()]
                        st.success(f"System started! Active models: {', '.join(active_models)}")

                except Exception as e:
                    st.error(f"Error initializing camera: {str(e)}")
                    st.session_state.running = False
            else:
                st.error("No models loaded successfully. Please check model paths.")

    # Handle stop button
    if stop_button and st.session_state.running:
        if st.session_state.manager:
            st.session_state.manager.stop_processing()
        if st.session_state.cap:
            st.session_state.cap.release()
        st.session_state.running = False
        st.success("System stopped successfully!")

    # Main interface
    if st.session_state.running and st.session_state.manager and st.session_state.cap:
        col1, col2 = st.columns([3, 1])

        with col1:
            st.subheader("📹 Live Feed")
            video_placeholder = st.empty()
            status_placeholder = st.empty()

        with col2:
            st.subheader("📊 Real-time Results")

            # Show active models
            active_models = list(st.session_state.manager.tflite_models.keys())
            for model in active_models:
                st.info(f"🔄 {model.capitalize()} Model Active")

            # Show current thresholds
            st.markdown("**Current Thresholds:**")
            st.write(f"Action: {st.session_state.manager.model_config['action']['threshold']}")
            st.write(f"Crowd: {st.session_state.manager.model_config['crowd']['threshold']}")

            # Real-time model results display
            results_placeholder = st.empty()

            st.subheader("🔔 Detection Log")
            message_placeholder = st.empty()

        # Performance tracking
        prev_time = 0
        frame_count = 0

        # Main processing loop
        while st.session_state.running:
            try:
                ret, frame = st.session_state.cap.read()
                if not ret:
                    st.error("Camera error - could not read frame")
                    break

                current_time = time.time()
                fps = 1 / (current_time - prev_time) if prev_time > 0 else 0
                prev_time = current_time
                frame_count += 1

                try:
                    st.session_state.manager.input_queue.put_nowait(frame.copy())
                except queue.Full:
                    pass

                results = None
                try:
                    results = st.session_state.manager.output_queue.get_nowait()
                except queue.Empty:
                    pass

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                if results:
                    detections = results.get("detections")
                    pose_landmarks = results.get("pose_landmarks")
                    frame_rgb = draw_predictions(frame_rgb, detections, pose_landmarks)

                cv2.putText(frame_rgb, f"FPS: {fps:.1f}", (10, frame_rgb.shape[0] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                video_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)

                # Enhanced status display
                active_detections = []
                if results and results.get("detections"):
                    for model_name, detection in results["detections"].items():
                        if detection:
                            active_detections.append(model_name)

                status_text = f"🟢 Running | Frame: {frame_count} | Models: {len(active_models)}"
                if active_detections:
                    status_text += f" | Active: {', '.join(active_detections)}"

                status_placeholder.success(status_text)

                # Display real-time model results in sidebar
                with results_placeholder.container():
                    if results and results.get("detections"):
                        detections = results["detections"]

                        # Action Recognition Results
                        if "action" in detections and detections["action"]:
                            action_data = detections["action"]
                            if action_data["class"] in ['Normal']:
                                st.success(f"🎯 **Action:** {action_data['class']} ({action_data['confidence']:.3f})")
                            elif action_data["class"] in ['Fighting', 'Assault', 'Abuse', 'Shooting']:
                                st.error(f"🚨 **Action:** {action_data['class']} ({action_data['confidence']:.3f})")
                            else:
                                st.warning(f"⚠️ **Action:** {action_data['class']} ({action_data['confidence']:.3f})")
                        else:
                            st.info("🎯 **Action:** No significant activity detected")

                        # Crowd Density Results
                        if "crowd" in detections and detections["crowd"]:
                            crowd_data = detections["crowd"]
                            density = crowd_data["density"]

                            if density > 2.0:
                                st.error(f"👥 **Crowd Density:** {density:.4f} (Very High)")
                            elif density > 1.0:
                                st.warning(f"👥 **Crowd Density:** {density:.4f} (High)")
                            elif density > 0.5:
                                st.info(f"👥 **Crowd Density:** {density:.4f} (Medium)")
                            else:
                                st.success(f"👥 **Crowd Density:** {density:.4f} (Low)")
                        else:
                            st.info("👥 **Crowd Density:** Below threshold")
                    else:
                        st.info("🎯 **Action:** No significant activity detected")
                        st.info("👥 **Crowd Density:** Below threshold")

                with message_placeholder.container():
                    display_message_history(st.session_state.manager.message_history)

                if stop_button:
                    break

                time.sleep(0.01)

            except Exception as e:
                st.error(f"Error in main loop: {str(e)}")
                break

    else:
        st.info("Configure your models and click 'Start' to begin detection")

        with st.expander("ℹ️ System Information"):
            st.markdown("""
            **This system can run both models simultaneously:**

            🎯 **Action Recognition Model:**
            - Detects various actions like fighting, theft, etc.
            - Adjustable confidence threshold
            - Real-time classification

            👥 **Crowd Density Model:**
            - Estimates crowd density in the scene
            - Handles different output formats (density maps/single values)
            - Adjustable density threshold

            **Performance Tips:**
            - Both models will process every frame (subject to frame skipping)
            - Check console output for detailed debug information
            - Adjust thresholds based on your specific use case
            - Lower frame skip for more responsive detection
            """)


if __name__ == "__main__":
    main()
