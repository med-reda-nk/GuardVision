import cv2
import numpy as np
from threading import Thread
import queue
import time
import tensorflow as tf
class ModelManager:
    def __init__(self):
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
        """Process a single frame through all loaded models"""
        results = {}
        for model_name, model in self.models.items():
            # Prepare input blob
            blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
            model.setInput(blob)
            
            # Get output layers
            layer_names = model.getLayerNames()
            output_layers = [layer_names[i - 1] for i in model.getUnconnectedOutLayers()]
            
            # Forward pass
            outputs = model.forward(output_layers)
            results[model_name] = outputs
            
        return results
    
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

# Example usage
if __name__ == "__main__":
    # Initialize model manager
    manager = ModelManager()
    
    # Load models
    manager.load_model("Action", "action_model.keras")
    manager.load_model("Crowd", "crowd_model.keras")
    manager.load_model("weapon","weapon_model.keras")
    # Start processing
    manager.start_processing()
    
    # Example of processing frames
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Add frame to processing queue
        manager.input_queue.put(frame)
        
        # Get results if available
        try:
            results = manager.output_queue.get_nowait()
            # Process results here
            print("Received results from models")
        except queue.Empty:
            pass
            
        # Display frame
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    # Cleanup
    manager.stop_processing()
    cap.release()
    cv2.destroyAllWindows()
