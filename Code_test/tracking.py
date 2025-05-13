import cv2
import mediapipe as mp
import time
import torch  # Importing PyTorch for YOLOv5

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


def main():
    pTime = 0
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    detector = PoseDetector()
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Load YOLOv5 model

    while True:
        success, img = cap.read()
        if not success:
            print("Error: Could not read frame.")
            break

        # Perform object detection
        results = model(img)
        detections = results.xyxy[0]  # Get detections

        for *box, conf, cls in detections:  # Iterate through detected objects
            if conf > 0.5:  # Confidence threshold
                x1, y1, x2, y2 = map(int, box)  # Get bounding box coordinates
                cropped_img = img[y1:y2, x1:x2]  # Crop the image

                # Detect pose in the cropped image
                cropped_img = detector.findPose(cropped_img)
                lmList = detector.findPosition(cropped_img)

                # Draw bounding box on the original image
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                if lmList:
                    print(f'Landmarks for detected person: {lmList}')

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, f'FPS: {int(fps)}', (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)

        cv2.imshow("Output", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()