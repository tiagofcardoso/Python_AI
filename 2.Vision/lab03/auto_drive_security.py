import os
import cv2
import numpy as np
from ultralytics import YOLO
import time
from collections import deque
import requests


class LaneDetector:
    def __init__(self):
        # Lane detection parameters
        self.roi_height = 0.6  # ROI height percentage from bottom
        self.min_line_length = 100
        self.max_line_gap = 50
        self.previous_lanes = deque(maxlen=5)  # Store previous lane detections

    def preprocess_frame(self, frame):
        """Preprocess frame for lane detection"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Edge detection
        edges = cv2.Canny(blurred, 50, 150)

        return edges

    def get_roi(self, frame):
        """Get region of interest for lane detection"""
        height, width = frame.shape[:2]
        roi_top = int(height * self.roi_height)

        # Define ROI polygon vertices
        vertices = np.array([
            [(0, height),
             (width * 0.45, roi_top),
             (width * 0.55, roi_top),
             (width, height)]
        ], dtype=np.int32)

        # Create mask
        mask = np.zeros_like(frame)
        cv2.fillPoly(mask, vertices, 255)

        # Apply mask
        masked = cv2.bitwise_and(frame, mask)
        return masked

    def detect_lanes(self, frame):
        """Detect lanes in the frame"""
        # Preprocess frame
        edges = self.preprocess_frame(frame)
        roi = self.get_roi(edges)

        # Detect lines using HoughLinesP
        lines = cv2.HoughLinesP(
            roi,
            rho=1,
            theta=np.pi/180,
            threshold=50,
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap
        )

        # Separate left and right lanes
        left_lanes = []
        right_lanes = []

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                slope = (y2 - y1) / (x2 - x1) if x2 != x1 else 0

                if slope < 0:  # Left lane
                    left_lanes.append(line[0])
                else:  # Right lane
                    right_lanes.append(line[0])

        return left_lanes, right_lanes


class TeslaVisionSystem:
    def __init__(self):
        # Initialize YOLO model for object detection
        self.model = YOLO('yolo11x-seg.pt')

        # Initialize lane detector
        self.lane_detector = LaneDetector()

        # Initialize detection history for tracking
        self.detection_history = deque(maxlen=10)

        # Define critical objects for emergency stopping
        self.critical_objects = ['person',
                                 'car',
                                 'motorcycle',
                                 'bicycle',
                                 'truck',
                                 'bus',
                                 'metro',
                                 'train',
                                 'traffic light',
                                 'stop sign']

        # Safety thresholds (in pixels)
        self.emergency_stop_threshold = 200
        self.warning_threshold = 400

        # Lane departure warning parameters
        self.lane_departure_threshold = 0.15  # Percentage of frame width

        # Initialize camera parameters
        self.camera = None
        self.frame_width = 1280
        self.frame_height = 720
        self.fps = 30

    def initialize_camera(self):
        # ip = "https://192.168.1.209:8080/video"
        # self.camera = cv2.VideoCapture(ip)
        #self.camera = cv2.VideoCapture(os.path.join(os.path.dirname(__file__), 'kart.mp4'))
        self.camera = cv2.VideoCapture(2)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        self.camera.set(cv2.CAP_PROP_FPS, self.fps)

        # Ignore SSL certificate warnings
        # requests.packages.urllib3.disable_warnings(
        # requests.packages.urllib3.exceptions.InsecureRequestWarning)

    def draw_lanes(self, frame, left_lanes, right_lanes):
        """Draw detected lanes on the frame"""
        lane_img = np.zeros_like(frame)

        # Draw left lanes
        if left_lanes:
            for lane in left_lanes:
                cv2.line(lane_img, (lane[0], lane[1]),
                         (lane[2], lane[3]), (0, 255, 0), 2)

        # Draw right lanes
        if right_lanes:
            for lane in right_lanes:
                cv2.line(lane_img, (lane[0], lane[1]),
                         (lane[2], lane[3]), (0, 255, 0), 2)

        # Combine with original frame
        return cv2.addWeighted(frame, 1, lane_img, 0.5, 0)

    def check_lane_departure(self, frame, left_lanes, right_lanes):
        """Check if vehicle is departing from lanes"""
        frame_center = self.frame_width / 2
        departure_warning = False
        departure_side = None

        if left_lanes and right_lanes:
            # Calculate average lane positions
            left_x = np.mean([lane[0] for lane in left_lanes])
            right_x = np.mean([lane[0] for lane in right_lanes])

            # Calculate center of lanes
            lane_center = (left_x + right_x) / 2

            # Check for departure
            if abs(frame_center - lane_center) > self.frame_width * self.lane_departure_threshold:
                departure_warning = True
                departure_side = "left" if lane_center > frame_center else "right"

        return departure_warning, departure_side

    def process_frame(self, frame):
        """Process a single frame for both object detection and lane detection"""
        # Object detection
        results = self.model(frame)
        emergency_stop = False
        warning = False

        # Process detected objects
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_name = result.names[int(box.cls[0])]
                conf = float(box.conf[0])

                if conf > 0.5 and class_name in self.critical_objects:
                    object_distance = self.frame_height - y2
                    if object_distance < self.emergency_stop_threshold:
                        emergency_stop = True
                    elif object_distance < self.warning_threshold:
                        warning = True

                    color = (0, 0, 255) if emergency_stop else (0, 255, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f'{class_name} {conf:.2f}',
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, color, 2)

        # Lane detection
        left_lanes, right_lanes = self.lane_detector.detect_lanes(frame)
        frame = self.draw_lanes(frame, left_lanes, right_lanes)

        # Check lane departure
        departure_warning, departure_side = self.check_lane_departure(
            frame, left_lanes, right_lanes)

        return frame, emergency_stop, warning, departure_warning, departure_side

    def run_vision_system(self):
        """Main loop for the vision system"""
        if not self.camera:
            self.initialize_camera()

        try:
            while True:
                ret, frame = self.camera.read()
                if not ret:
                    break

                # Process the frame
                processed_frame, emergency_stop, warning, departure_warning, departure_side = self.process_frame(
                    frame)

                # Display status
                status_messages = []
                if emergency_stop:
                    status_messages.append(("EMERGENCY STOP!", (0, 0, 255)))
                elif warning:
                    status_messages.append(("WARNING!", (0, 255, 255)))
                else:
                    status_messages.append(("Safe", (0, 255, 0)))

                if departure_warning:
                    status_messages.append(
                        (f"Lane Departure: {departure_side}", (255, 165, 0)))

                # Draw status messages
                for i, (msg, color) in enumerate(status_messages):
                    cv2.putText(processed_frame, msg,
                                (10, 30 + (i * 30)), cv2.FONT_HERSHEY_SIMPLEX,
                                1, color, 2)

                # Display the frame
                cv2.imshow('Tesla Vision System', processed_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                # Vehicle control
                if emergency_stop:
                    self.emergency_stop_procedure()
                elif warning:
                    self.warning_procedure()
                elif departure_warning:
                    self.lane_departure_procedure(departure_side)

        finally:
            self.cleanup()

    def emergency_stop_procedure(self):
        """Simulate emergency stop procedure"""
        print("EMERGENCY STOP INITIATED!")
        time.sleep(0.1)

    def warning_procedure(self):
        """Simulate warning procedure"""
        print("Warning: Object detected in warning zone")
        time.sleep(0.1)

    def lane_departure_procedure(self, side):
        """Simulate lane departure warning procedure"""
        print(f"Lane Departure Warning: Drifting to {side}")
        time.sleep(0.1)

    def cleanup(self):
        """Clean up resources"""
        if self.camera:
            self.camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    vision_system = TeslaVisionSystem()
    vision_system.run_vision_system()
