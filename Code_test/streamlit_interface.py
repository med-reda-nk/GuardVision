import streamlit as st
import cv2
import numpy as np
import time
import queue
import tensorflow as tf
import os
from threading import Thread, Lock
from collections import deque
import mediapipe as mp
import torch
from datetime import datetime, timedelta
import json
from collections import Counter, defaultdict
import re
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Disable oneDNN optimizations for reproducibility
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Local model path for EfficientDet
MODEL_PATH = 'models/crowd_model'  # Ensure this matches your directory structure

class NLPReportGenerator:
    def __init__(self):
        self.daily_events = defaultdict(list)
        self.session_start_time = datetime.now()
        self.threat_keywords = {
            'high_threat': ['shooting', 'assault', 'fighting', 'abuse', 'explosion', 'arson'],
            'medium_threat': ['robbery', 'burglary', 'stealing', 'shoplifting', 'vandalism'],
            'crowd_related': ['crowd', 'density', 'gathering']
        }
        self.people_count_stats = {
            'max_people': 0,
            'total_detections': 0,
            'average_people': 0
        }

    def log_event(self, event_type, details, timestamp=None):
        if timestamp is None:
            timestamp = datetime.now()

        people_count = 0
        if 'people' in details.lower():
            import re
            numbers = re.findall(r'\d+', details)
            if numbers:
                people_count = int(numbers[0])

        event_data = {
            'timestamp': timestamp,
            'type': event_type,
            'details': details,
            'hour': timestamp.hour,
            'people_count': people_count
        }

        # Update people count statistics
        if people_count > 0:
            self.people_count_stats['max_people'] = max(self.people_count_stats['max_people'], people_count)
            self.people_count_stats['total_detections'] += 1

            # Calculate running average
            if self.people_count_stats['total_detections'] > 0:
                total_people = sum(event.get('people_count', 0) for events in self.daily_events.values()
                                   for event in events if event.get('people_count', 0) > 0)
                self.people_count_stats['average_people'] = total_people / self.people_count_stats['total_detections']

        date_key = timestamp.strftime('%Y-%m-%d')
        self.daily_events[date_key].append(event_data)

    def generate_daily_summary(self, target_date=None):
        if target_date is None:
            target_date = datetime.now().strftime('%Y-%m-%d')
        events = self.daily_events.get(target_date, [])
        if not events:
            return "No significant events recorded for today."
        analysis = self._analyze_events(events)
        report = self._generate_nlp_report(analysis, target_date)
        return report

    def _analyze_events(self, events):
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
        if events:
            start_time = min(event['timestamp'] for event in events)
            end_time = max(event['timestamp'] for event in events)
            analysis['duration_minutes'] = int((end_time - start_time).total_seconds() / 60)
        for event in events:
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
            if any(keyword in event_detail for keyword in self.threat_keywords['crowd_related']):
                analysis['crowd_events'] += 1
            analysis['hourly_distribution'][event['hour']] += 1
            analysis['event_types'][event['type']] += 1
        if analysis['hourly_distribution']:
            max_events = max(analysis['hourly_distribution'].values())
            analysis['peak_hours'] = [hour for hour, count in analysis['hourly_distribution'].items()
                                    if count == max_events]
        return analysis

    def _generate_nlp_report(self, analysis, date):
        report_parts = []
        report_parts.append(f"📊 DAILY SURVEILLANCE REPORT - {date}")
        report_parts.append("=" * 50)
        total_events = analysis['total_events']
        duration = analysis['duration_minutes']
        if total_events == 0:
            return "No security events detected during the monitoring period."
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
        report_parts.append(f"\nTHREAT ASSESSMENT: {threat_status}")
        report_parts.append(f"SUMMARY: {summary}")
        report_parts.append(f"\nOPERATIONAL METRICS:")
        report_parts.append(f"   • Monitoring Duration: {duration} minutes")
        report_parts.append(f"   • Total Events Detected: {total_events}")
        report_parts.append(f"   • Events per Hour: {total_events / max(1, duration / 60):.1f}")

        if self.people_count_stats['max_people'] > 0:
            report_parts.append(f"\n👥 PEOPLE COUNT ANALYSIS:")
            report_parts.append(f"   • Maximum People Detected: {self.people_count_stats['max_people']}")
            report_parts.append(f"   • Average People Count: {self.people_count_stats['average_people']:.1f}")
            report_parts.append(f"   • Total People Detections: {self.people_count_stats['total_detections']}")

            # Occupancy level assessment
            max_people = self.people_count_stats['max_people']
            if max_people > 20:
                occupancy_level = "Very High Occupancy"
            elif max_people > 10:
                occupancy_level = "High Occupancy"
            elif max_people > 5:
                occupancy_level = "Moderate Occupancy"
            else:
                occupancy_level = "Low Occupancy"

            report_parts.append(f"   • Occupancy Assessment: {occupancy_level}")

        if analysis['peak_hours']:
            peak_times = [f"{hour:02d}:00" for hour in analysis['peak_hours']]
            report_parts.append(f"   • Peak Activity Hours: {', '.join(peak_times)}")
        if high_threats > 0 or medium_threats > 0:
            report_parts.append(f"\n⚠️  THREAT BREAKDOWN:")
            if high_threats > 0:
                report_parts.append(f"   • High Priority Threats: {high_threats}")
            if medium_threats > 0:
                report_parts.append(f"   • Medium Priority Threats: {medium_threats}")
        if analysis['crowd_events'] > 0:
            crowd_percentage = (analysis['crowd_events'] / total_events) * 100
            report_parts.append(f"\n👥 CROWD ANALYSIS:")
            report_parts.append(f"   • Crowd-related Events: {analysis['crowd_events']} ({crowd_percentage:.1f}%)")
        if analysis['threat_summary']:
            report_parts.append(f"\n🚨 CRITICAL EVENTS LOG:")
            for threat in analysis['threat_summary'][:5]:
                report_parts.append(f"   • {threat}")
            if len(analysis['threat_summary']) > 5:
                report_parts.append(f"   • ... and {len(analysis['threat_summary']) - 5} more events")
        recommendations = self._generate_recommendations(analysis)
        if recommendations:
            report_parts.append(f"\nRECOMMENDATIONS:")
            for rec in recommendations:
                report_parts.append(f"   • {rec}")
        return "\n".join(report_parts)

    def _generate_recommendations(self, analysis):
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

def display_daily_report_section(manager):
    if manager is None or not hasattr(manager, 'nlp_reporter'):
        st.info("No report available. Start the system to generate reports.")
        return
    st.subheader("Daily Report")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Generate Report", key="generate_report_button"):
            report = manager.nlp_reporter.generate_daily_summary()
            st.session_state.daily_report = report
    with col2:
        if st.button("Save Report", key="save_report_button"):
            if hasattr(st.session_state, 'daily_report'):
                filename = f"surveillance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                st.download_button(
                    label="Download Report",
                    data=st.session_state.daily_report,
                    file_name=filename,
                    mime="text/plain"
                )
    if hasattr(st.session_state, 'daily_report'):
        with st.expander("View Report", expanded=False):
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
        self.saved_models = {}
        self.input_queue = queue.Queue(maxsize=3)
        self.output_queue = queue.Queue(maxsize=3)
        self.running = False
        self.last_prediction = None
        self.message_history = deque(maxlen=20)
        self.frame_skip_counter = 0
        self.process_every_n_frames = 1
        self.detection_lock = Lock()
        self.last_detections = {
            "action": None,
            "crowd": None
        }
        self.last_people_count = 0
        self.people_count_threshold = 1
        self.model_config = {
            "action": {
                "threshold": 0.7,
                "class_names": ['Abuse', 'Arrest', 'Arson', 'Assault', 'Burglary', 'Explosion', 'Fighting', "Normal",
                                'Robbery', 'Shooting', 'Shoplifting', 'Stealing', 'Vandalism']
            },
            "crowd": {
                "threshold": 0.5,
                "density": 0
            }
        }
        self.pose_detector = OptimizedPoseDetector()
        try:
            self.yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5n')
            self.yolo_model.conf = 0.6
            self.yolo_model.iou = 0.45
            self.yolo_model.classes = [0]
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.yolo_model.to(self.device)
            logger.info(f"Using device: {self.device}")
        except Exception as e:
            logger.error(f"Error loading YOLO model: {str(e)}")
            self.yolo_model = None

    def load_model(self, model_name, model_path):
        try:
            if model_name == "action":
                if not os.path.exists(model_path):
                    st.error(f"Action model not found at: {model_path}")
                    logger.error(f"Action model not found at: {model_path}")
                    return False
                model = tf.keras.models.load_model(model_path)
                logger.info(f"\n=== {model_name} Model Info ===")
                logger.info(f"Input shape: {model.input_shape}")
                logger.info(f"Output shape: {model.output_shape}")
                model.summary(print_fn=lambda x: logger.info(x))
                converter = tf.lite.TFLiteConverter.from_keras_model(model)
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                tflite_model = converter.convert()
                interpreter = tf.lite.Interpreter(model_content=tflite_model)
                interpreter.allocate_tensors()
                input_details = interpreter.get_input_details()
                output_details = interpreter.get_output_details()
                logger.info(f"\n=== {model_name} TFLite Details ===")
                logger.info(f"Input details: {input_details}")
                logger.info(f"Output details: {output_details}")
                self.tflite_models[model_name] = {
                    'interpreter': interpreter,
                    'input_details': input_details,
                    'output_details': output_details
                }
            elif model_name == "crowd":
                if not os.path.exists(model_path):
                    st.error(f"Crowd model directory not found at: {model_path}")
                    logger.error(f"Crowd model directory not found at: {model_path}")
                    return False
                logger.info(f"Attempting to load SavedModel from {model_path}")
                model = tf.saved_model.load(model_path)
                # Verify the model signature
                signature = model.signatures["serving_default"]
                logger.info(f"Model signatures: {list(model.signatures.keys())}")
                logger.info(f"Signature inputs: {signature.structured_input_signature}")
                logger.info(f"Signature outputs: {signature.structured_outputs}")
                self.saved_models[model_name] = model
            st.success(f"{model_name} successfully loaded")
            return True
        except Exception as e:
            error_msg = f"Failed to load {model_name}: {str(e)}"
            st.error(error_msg)
            logger.error(error_msg, exc_info=True)
            return False

    def process_frame(self, frame):
        self.frame_skip_counter += 1
        if self.frame_skip_counter % self.process_every_n_frames != 0:
            return None

        small_frame = cv2.resize(frame, (320, 240))
        detections = {}
        pose_landmarks = []
        pose_people_count = 0  # Add pose-based people counter

        # Action detection (existing code remains the same)
        if "action" in self.tflite_models:
            try:
                interpreter = self.tflite_models["action"]['interpreter']
                input_details = self.tflite_models["action"]['input_details']
                output_details = self.tflite_models["action"]['output_details']

                input_shape = input_details[0]['shape']
                expected_height, expected_width = input_shape[1], input_shape[2]
                frame_resized = cv2.resize(frame, (expected_width, expected_height))
                blob = frame_resized.astype(np.float32) / 255.0
                blob = np.expand_dims(blob, axis=0)

                interpreter.set_tensor(input_details[0]['index'], blob)
                interpreter.invoke()
                preds = interpreter.get_tensor(output_details[0]['index'])

                if len(preds.shape) > 1:
                    preds = preds[0]

                class_id = np.argmax(preds)
                confidence = preds[class_id]

                if confidence > self.model_config["action"]["threshold"]:
                    class_name = self.model_config["action"]["class_names"][class_id]
                    with self.detection_lock:
                        detections["action"] = {
                            "class": class_name,
                            "confidence": float(confidence)
                        }
                    self._add_detection_message("action", class_name, confidence)
                    logger.info(f"ACTION DETECTED: {class_name} ({confidence:.3f})")

            except Exception as e:
                logger.error(f"TFLite prediction error in action model: {str(e)}", exc_info=True)

        # Enhanced pose detection for people counting
        try:
            # First, try to detect poses directly on the full frame
            pose_frame = self.pose_detector.findPose(frame.copy(), draw=False)
            full_frame_landmarks = self.pose_detector.findPosition(pose_frame, draw=False)

            if full_frame_landmarks:
                pose_landmarks.append(full_frame_landmarks)
                pose_people_count += 1

            # If YOLO is available, use it for better person detection
            if self.yolo_model is not None:
                try:
                    with torch.amp.autocast('cuda'):
                        results = self.yolo_model(small_frame, size=320)
                        yolo_detections = results.xyxy[0].cpu().numpy()

                    scale_x = frame.shape[1] / small_frame.shape[1]
                    scale_y = frame.shape[0] / small_frame.shape[0]

                    yolo_people_count = 0


                    for detection in yolo_detections:


                        x1, y1, x2, y2, conf, cls = detection
                        if conf > 0.5 and cls == 0:  # Person class
                            yolo_people_count += 1

                            # Scale coordinates back to original frame
                            x1, y1, x2, y2 = int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y)
                            x1, y1 = max(0, x1), max(0, y1)
                            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

                            if x2 > x1 and y2 > y1:
                                cropped_img = frame[y1:y2, x1:x2]
                                if cropped_img.shape[0] > 50 and cropped_img.shape[1] > 50:
                                    # Try to detect pose in cropped region
                                    cropped_img = self.pose_detector.findPose(cropped_img, draw=False)
                                    lmList = self.pose_detector.findPosition(cropped_img, draw=False)

                                    if lmList:
                                        # Adjust landmarks to original frame coordinates
                                        adjusted_landmarks = []
                                        for id, cx, cy in lmList:
                                            original_cx = cx + x1
                                            original_cy = cy + y1
                                            adjusted_landmarks.append([id, original_cx, original_cy])
                                        pose_landmarks.append(adjusted_landmarks)

                    # Use YOLO count if it's higher (more reliable for multiple people)
                    if yolo_people_count > pose_people_count:
                        pose_people_count = yolo_people_count

                    logger.debug(
                        f"YOLO detected {yolo_people_count} people, pose detected {len(pose_landmarks)} people")

                except Exception as e:
                    logger.error(f"YOLO processing error: {str(e)}", exc_info=True)

        except Exception as e:
            logger.error(f"Pose detection error: {str(e)}", exc_info=True)

        if "action" in self.tflite_models and detections.get("action"):
            action_data = detections["action"]
            self._add_detection_message("action", action_data["class"],
                                        action_data["confidence"], pose_people_count)

        # Crowd detection (existing code with pose integration)
        if "crowd" in self.saved_models:
            try:
                model = self.saved_models["crowd"]
                input_shape = (512, 512)
                frame_resized = cv2.resize(frame, input_shape)
                input_tensor = tf.convert_to_tensor(frame_resized[None, ...], dtype=tf.uint8)

                signature = model.signatures["serving_default"]
                detections_output = signature(input_tensor)

                boxes = detections_output['detection_boxes'][
                    0].numpy() if 'detection_boxes' in detections_output else None
                scores = detections_output['detection_scores'][
                    0].numpy() if 'detection_scores' in detections_output else None
                classes = detections_output['detection_classes'][0].numpy().astype(
                    np.int32) if 'detection_classes' in detections_output else None

                if boxes is not None and scores is not None and classes is not None:
                    threshold = self.model_config["crowd"]["threshold"]
                    person_mask = (classes == 1) & (scores > threshold)
                    crowd_person_count = np.sum(person_mask)
                    density = crowd_person_count / (frame_resized.shape[0] * frame_resized.shape[1] / 10000)

                    # Combine crowd detection with pose count
                    final_person_count = max(crowd_person_count, pose_people_count)

                    if density > self.model_config["crowd"]["threshold"] or pose_people_count > 0:
                        with self.detection_lock:
                            detections["crowd"] = {
                                "density": density,
                                "person_count": int(final_person_count),
                                "pose_count": pose_people_count,  # Add pose-specific count
                                "crowd_model_count": int(crowd_person_count)
                            }
                        self._add_detection_message("crowd", None, density, final_person_count)
                        logger.info(
                            f"PEOPLE DETECTED: Pose={pose_people_count}, Crowd Model={crowd_person_count}, Final={final_person_count}")

            except Exception as e:
                logger.error(f"SavedModel prediction error: {str(e)}", exc_info=True)

        # If no crowd model but pose detected people
        elif pose_people_count != self.last_people_count:
            self._add_detection_message("people_count", None, None, pose_people_count)
            detections["crowd"] = {
                "density": pose_people_count * 0.1,  # Simple density estimation
                "person_count": pose_people_count,
                "pose_count": pose_people_count,
                "crowd_model_count": 0
            }

        with self.detection_lock:
            if detections:
                self.last_detections.update(detections)

        return {
            "detections": detections if detections else None,
            "pose_landmarks": pose_landmarks if pose_landmarks else None,
            "people_count": pose_people_count  # Add this for easy access
        }

    def initialize_nlp_reporter(self):
        self.nlp_reporter = NLPReportGenerator()

    def _add_detection_message(self, model_name, class_name, value, people_count=None):
        timestamp = time.strftime("%H:%M:%S")
        current_time = datetime.now()
        with self.detection_lock:
            if model_name == "action":
                if (self.last_detections.get("action") is None or
                        self.last_detections["action"].get("class") != class_name):
                    if people_count is not None and people_count > 0:
                        message = f"🚨 [{timestamp}] Action: {class_name} (Conf: {value:.2f}) | People: {people_count}"
                    else:
                        message = f"🚨 [{timestamp}] Action: {class_name} (Conf: {value:.2f})"
                    self.message_history.append(message)
                    if hasattr(self, 'nlp_reporter'):
                        event_details = f"{class_name} detected with {value:.2f} confidence"
                        if people_count is not None and people_count > 0:
                            event_details += f", {people_count} people present"

                        self.nlp_reporter.log_event(
                            event_type="action_detection",
                            details=event_details,
                            timestamp=current_time
                        )

            elif model_name == "crowd":
                        # Check if people count changed significantly
                people_count_changed = (people_count is not None and
                                                abs(people_count - self.last_people_count) >= self.people_count_threshold)
                        # Check if density changed significantly
                density_changed = (self.last_detections.get("crowd") is None or
                                           abs(self.last_detections.get("crowd", {}).get("density", 0) - value) > 0.1)
                if people_count_changed or density_changed:
                    if people_count is not None and people_count > 0:
                        if people_count > 10:
                            message = f"👥 [{timestamp}] High Crowd: {people_count} people (Density: {value:.3f})"
                        elif people_count > 5:
                            message = f"👥 [{timestamp}] Moderate Crowd: {people_count} people (Density: {value:.3f})"
                        elif people_count > 2:
                            message = f"👥 [{timestamp}] Small Group: {people_count} people (Density: {value:.3f})"
                        else:
                            message = f"👥 [{timestamp}] People Detected: {people_count} (Density: {value:.3f})"
                    else:
                        message = f"👥 [{timestamp}] Crowd density: {value:.4f}"
                    self.message_history.append(message)
                    self.last_people_count = people_count if people_count is not None else 0
                    if hasattr(self, 'nlp_reporter'):
                        event_details = f"Crowd density level {value:.4f}"
                        if people_count is not None and people_count > 0:
                            event_details = f"{people_count} people detected, density level {value:.4f}"
                        self.nlp_reporter.log_event(
                            event_type="crowd_detection",
                            details=event_details,
                            timestamp=current_time
                        )

            elif model_name == "people_count":
                        # New logging category specifically for people count changes
                if (people_count is not None and
                                abs(people_count - self.last_people_count) >= self.people_count_threshold):
                    if people_count == 0:
                        message = f"👤 [{timestamp}] Area Clear: No people detected"
                    elif people_count == 1:
                        message = f"👤 [{timestamp}] Single Person: 1 person detected"
                    else:
                        message = f"👥 [{timestamp}] Multiple People: {people_count} people detected"
                    self.message_history.append(message)
                    self.last_people_count = people_count
                    if hasattr(self, 'nlp_reporter'):
                        if people_count == 0:
                            event_details = "Area cleared - no people detected"
                        else:
                            event_details = f"{people_count} people detected in monitoring area"
                        self.nlp_reporter.log_event(
                            event_type="people_count",
                            details=event_details,
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
                if results:
                    try:
                        self.output_queue.put_nowait(results)
                    except queue.Full:
                        pass
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Processing error: {str(e)}", exc_info=True)

def draw_predictions(frame, detections, pose_landmarks):
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
    """Enhanced message history display with better formatting"""
    if not messages:
        st.info("System running - no alerts detected")
    else:
        recent_messages = list(messages)[-15:]  # Show more messages

        # Group messages by type for better organization
        action_messages = []
        crowd_messages = []
        people_messages = []
        other_messages = []

        for msg in reversed(recent_messages):
            if "🚨" in msg and "Action" in msg:
                action_messages.append(msg)
            elif "👥" in msg and ("Crowd" in msg or "Group" in msg):
                crowd_messages.append(msg)
            elif "👤" in msg or ("👥" in msg and "People" in msg):
                people_messages.append(msg)
            else:
                other_messages.append(msg)

        # Display messages by category
        if action_messages:
            st.markdown("**🚨 Action Alerts:**")
            for msg in action_messages[:5]:  # Show last 5 action alerts
                st.error(msg)

        if crowd_messages:
            st.markdown("**👥 Crowd Alerts:**")
            for msg in crowd_messages[:5]:  # Show last 5 crowd alerts
                st.warning(msg)

        if people_messages:
            st.markdown("**👤 People Count:**")
            for msg in people_messages[:3]:  # Show last 3 people count updates
                st.info(msg)

        if other_messages:
            for msg in other_messages:
                st.info(msg)

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
        .css-1aumxhk {  /* Adjust image width using CSS */
            max-width: 100%;
        }
        </style>
    """, unsafe_allow_html=True)
    st.title("GuardVision")
    st.markdown("**Surveillance System**")
    if 'manager' not in st.session_state:
        st.session_state.manager = None
        st.session_state.cap = None
        st.session_state.running = False
    with st.sidebar:
        st.header("Control Panel")
        use_action = st.checkbox("Action Recognition", value=True)
        use_crowd = st.checkbox("Crowd Density", value=True)
        st.subheader("Model Thresholds")
        action_threshold = st.slider("Action Confidence Threshold", 0.1, 1.0, 0.7, 0.05)
        crowd_threshold = st.slider("Crowd Density Threshold", 0.1, 2.0, 0.5, 0.1)
        st.subheader("Performance Settings")
        frame_skip = st.slider("Frame Skip", 1, 10, 3)
        st.subheader("Camera Settings")
        camera_resolution = st.selectbox("Resolution", ["320x240", "640x480", "1280x720"], index=1)
        st.subheader("Model Status")
        display_daily_report_section(st.session_state.manager)
        if st.session_state.manager and (st.session_state.manager.tflite_models or st.session_state.manager.saved_models):
            for model_name in list(st.session_state.manager.tflite_models.keys()) + list(st.session_state.manager.saved_models.keys()):
                st.success(f"✅ {model_name.capitalize()} Model Loaded")
        col1, col2 = st.columns(2)
        with col1:
            start_button = st.button("Start", type="primary")
        with col2:
            stop_button = st.button("Stop", type="secondary")
    if start_button and not st.session_state.running:
        if not use_action and not use_crowd:
            st.error("Please select at least one model to use!")
        else:
            st.session_state.manager = OptimizedModelManager()
            st.session_state.manager.initialize_nlp_reporter()
            st.session_state.manager.model_config["action"]["threshold"] = action_threshold
            st.session_state.manager.model_config["crowd"]["threshold"] = crowd_threshold
            st.session_state.manager.process_every_n_frames = frame_skip
            models_loaded = False
            models_to_load = []
            if use_action:
                models_to_load.append(("action", "models/action_model.keras"))
            if use_crowd:
                models_to_load.append(("crowd", MODEL_PATH))
            with st.spinner("Loading models..."):
                for model_name, model_path in models_to_load:
                    if st.session_state.manager.load_model(model_name, model_path):
                        models_loaded = True
                    else:
                        st.error(f"Failed to load {model_name} model from {model_path}. Check console logs for details.")
            if models_loaded:
                st.session_state.manager.start_processing()
                try:
                    st.session_state.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Switch to DirectShow backend
                    if not st.session_state.cap.isOpened():
                        st.error("Could not open camera. Please ensure no other application is using it and run as administrator.")
                        logger.error("Failed to open camera with DirectShow backend")
                        st.session_state.running = False
                    else:
                        width, height = map(int, camera_resolution.split('x'))
                        st.session_state.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                        st.session_state.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                        st.session_state.cap.set(cv2.CAP_PROP_FPS, 30)
                        st.session_state.running = True
                        active_models = list(st.session_state.manager.tflite_models.keys()) + list(st.session_state.manager.saved_models.keys())
                        st.success(f"System started! Active models: {', '.join(active_models)}")
                except Exception as e:
                    st.error(f"Error initializing camera: {str(e)}")
                    logger.error(f"Camera initialization error: {str(e)}", exc_info=True)
                    st.session_state.running = False
            else:
                st.error("No models loaded successfully. Please check model paths and console logs.")
    if stop_button and st.session_state.running:
        if st.session_state.manager:
            st.session_state.manager.stop_processing()
        if st.session_state.cap:
            st.session_state.cap.release()
            st.session_state.cap = None
        st.session_state.running = False
        st.success("System stopped successfully!")
    if st.session_state.running and st.session_state.manager and st.session_state.cap:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.subheader("Live Feed")
            video_placeholder = st.empty()
            status_placeholder = st.empty()
        with col2:
            st.subheader("Real-time Results")
            active_models = list(st.session_state.manager.tflite_models.keys()) + list(st.session_state.manager.saved_models.keys())
            for model in active_models:
                st.info(f"🔄 {model.capitalize()} Model Active")
            st.markdown("**Current Thresholds:**")
            st.write(f"Action: {st.session_state.manager.model_config['action']['threshold']}")
            st.write(f"Crowd: {st.session_state.manager.model_config['crowd']['threshold']}")
            results_placeholder = st.empty()
            st.subheader("Detection Log")
            message_placeholder = st.empty()
        prev_time = 0
        frame_count = 0
        try:
            while st.session_state.running:
                ret, frame = st.session_state.cap.read()
                if not ret:
                    st.error("Camera error - failed to grab frame. Check camera connection or permissions.")
                    logger.warning("Failed to grab frame from camera")
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
                    people_count = results.get("people_count", 0)
                    frame_rgb = draw_predictions(frame_rgb, detections, pose_landmarks)
                cv2.putText(frame_rgb, f"FPS: {fps:.1f}", (10, frame_rgb.shape[0] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                video_placeholder.image(frame_rgb, channels="RGB", width=800)  # Fixed width value
                active_detections = []
                if results and results.get("detections"):
                    for model_name, detection in results["detections"].items():
                        if detection:
                            active_detections.append(model_name)
                status_text = f"🟢 Running | Frame: {frame_count} | Models: {len(active_models)}"
                if active_detections:
                    status_text += f" | Active: {', '.join(active_detections)}"
                status_placeholder.success(status_text)
                with results_placeholder.container():
                    if results and results.get("detections"):
                        detections = results["detections"]

                        if "action" in detections and detections["action"]:
                            action_data = detections["action"]
                            if action_data["class"] in ['Normal']:
                                st.success(f"🎯 **Action:** {action_data['class']} ({action_data['confidence']:.3f})")
                            elif action_data["class"] in ['Fighting', 'Assault', 'Abuse', 'Shooting']:
                                st.error(f"🚨 **Action:** {action_data['class']} ({action_data['confidence']:.3f})")
                            else:
                                st.warning(f"⚠️ **Action:** {action_data['class']} ({action_data['confidence']:.3f})")
                        else:
                            time.sleep(1)
                            st.info("🎯 **Action:** No significant activity detected")

                        if "crowd" in detections and detections["crowd"]:
                            crowd_data = detections["crowd"]
                            total_people = crowd_data["person_count"]
                            pose_count = crowd_data.get("pose_count", 0)
                            crowd_model_count = crowd_data.get("crowd_model_count", 0)
                            density = crowd_data["density"]

                            # Create detailed people count display
                            people_info = f"👥 **People Count:** {total_people} "
                            if pose_count > 0 and crowd_model_count > 0:
                                people_info += f"(Pose: {pose_count}, Crowd Model: {crowd_model_count})"
                            elif pose_count > 0:
                                people_info += f"(Detected via Pose Analysis)"
                            elif crowd_model_count > 0:
                                people_info += f"(Detected via Crowd Model)"

                            # Display with appropriate color based on count
                            if total_people > 10:
                                st.error(f"{people_info} - Very Crowded")
                            elif total_people > 5:
                                st.warning(f"{people_info} - Crowded")
                            elif total_people > 2:
                                st.info(f"{people_info} - Moderate")
                            elif total_people > 0:
                                st.success(f"{people_info} - Light")

                            # Also show density if available
                            if density > 0:
                                st.write(f"   Density Score: {density:.4f}")
                        else:
                            people_count = results.get("people_count", 0)
                            if people_count > 0:
                                st.success(f"👥 **People Count:** {people_count} (Detected via Pose Analysis)")
                            time.sleep(1)
                            st.info("👥 **Crowd Density:** Below threshold")

                    else:
                        time.sleep(1)
                        st.info("🎯 **Action:** No significant activity detected")
                        st.info("👥 **Crowd Density:** Below threshold")
                with message_placeholder.container():
                    display_message_history(st.session_state.manager.message_history)
                if stop_button:
                    break
                time.sleep(0.01)
        except Exception as e:
            st.error(f"Error in main loop: {str(e)}")
            logger.error(f"Main loop error: {str(e)}", exc_info=True)
        finally:
            if st.session_state.cap:
                st.session_state.cap.release()
                st.session_state.cap = None
            if st.session_state.manager:
                st.session_state.manager.stop_processing()
            st.session_state.running = False
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
            - Uses EfficientDet to detect persons and estimate crowd density
            - Adjustable density threshold
            - Real-time person counting

            **Performance Tips:**
            - Both models will process every frame (subject to frame skipping)
            - Check console output for detailed debug information
            - Adjust thresholds based on your specific use case
            - Lower frame skip for more responsive detection
            """)

if __name__ == "__main__":
    main()
