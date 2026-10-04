from __future__ import annotations

import cv2
import numpy as np


class DroneTargetDetector:
    def __init__(self, source, mode, target_color="red", model_path="yolov8n.pt",
                 conf_threshold=0.5, output_path=None, debug=False):
        self.source = source
        self.mode = mode
        self.target_color = target_color.lower()
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.output_path = output_path
        self.debug = debug
        self.cap = None
        self.writer = None
        self.model = None

        if self.mode == "yolo":
            self._load_yolo_model()

    def _load_yolo_model(self):
        try:
            from ultralytics import YOLO
            self.model = YOLO(self.model_path)
        except Exception as exc:
            raise RuntimeError(f"YOLO model could not be loaded: {exc}") from exc

    def _get_color_bounds(self):
        palette = {
            "red": [(np.array([0, 100, 100]), np.array([10, 255, 255])),
                    (np.array([160, 100, 100]), np.array([180, 255, 255]))],
            "green": [(np.array([35, 50, 50]), np.array([90, 255, 255]))],
            "blue": [(np.array([90, 50, 50]), np.array([130, 255, 255]))],
            "yellow": [(np.array([20, 80, 80]), np.array([40, 255, 255]))],
            "orange": [(np.array([5, 80, 80]), np.array([25, 255, 255]))],
            "purple": [(np.array([125, 50, 50]), np.array([160, 255, 255]))],
        }
        if self.target_color not in palette:
            raise ValueError(f"Unsupported color: {self.target_color}")
        return palette[self.target_color]

    def _detect_color_target(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        bounds = self._get_color_bounds()
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)

        for lower, upper in bounds:
            mask += cv2.inRange(hsv, lower, upper)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        if area < 400:
            return None

        x, y, w, h = cv2.boundingRect(largest)
        return {"x": x, "y": y, "w": w, "h": h, "area": area, "label": "color_target"}

    def _detect_circle_target(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (9, 9), 2)
        circles = cv2.HoughCircles(
            blur,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=50,
            param1=100,
            param2=30,
            minRadius=10,
            maxRadius=200,
        )

        if circles is None:
            return None

        circles = np.round(circles[0, :]).astype("int")
        best = None
        for (x, y, r) in circles:
            area = np.pi * r * r
            if best is None or area > best["area"]:
                best = {"x": x - r, "y": y - r, "w": 2 * r, "h": 2 * r, "area": area, "label": "circle_target"}
        return best

    def _detect_yolo_target(self, frame):
        if self.model is None:
            return None

        results = self.model(frame, conf=self.conf_threshold, verbose=False)
        if not results or len(results[0].boxes) == 0:
            return None

        boxes = results[0].boxes
        confs = boxes.conf.tolist()
        best_idx = max(range(len(confs)), key=lambda i: confs[i])
        box = boxes[best_idx]
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        label = results[0].names[int(box.cls[0])]

        return {
            "x": int(x1),
            "y": int(y1),
            "w": int(x2 - x1),
            "h": int(y2 - y1),
            "area": int((x2 - x1) * (y2 - y1)),
            "label": label,
        }

    def _compute_target_status(self, frame, detection):
        height, width = frame.shape[:2]
        cx = width // 2
        cy = height // 2

        target_cx = detection["x"] + detection["w"] // 2
        target_cy = detection["y"] + detection["h"] // 2

        dx = target_cx - cx
        dy = target_cy - cy

        if abs(dx) < 25 and abs(dy) < 25:
            status = "centered"
        else:
            if abs(dx) > abs(dy):
                status = "move right" if dx > 0 else "move left"
            else:
                status = "move down" if dy > 0 else "move up"

        decision = {
            "image_center": (cx, cy),
            "target_center": (target_cx, target_cy),
            "dx": dx,
            "dy": dy,
            "status": status,
        }
        return decision

    def _draw_detection(self, frame, detection, decision):
        x, y, w, h = detection["x"], detection["y"], detection["w"], detection["h"]
        label = detection["label"]

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(frame, (x + w // 2, y + h // 2), 5, (0, 0, 255), -1)
        cv2.putText(frame, f"{label}", (x, max(y - 10, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        height, width = frame.shape[:2]
        cx = width // 2
        cy = height // 2
        cv2.line(frame, (cx, 0), (cx, height), (255, 255, 255), 1)
        cv2.line(frame, (0, cy), (width, cy), (255, 255, 255), 1)

        cv2.putText(frame, f"dx={decision['dx']} dy={decision['dy']}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, decision["status"], (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    def run(self):
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open source: {self.source}")

        if self.output_path:
            fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.writer = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            if self.mode == "color":
                detection = self._detect_color_target(frame)
            elif self.mode == "circle":
                detection = self._detect_circle_target(frame)
            elif self.mode == "yolo":
                detection = self._detect_yolo_target(frame)
            else:
                detection = None

            if detection is not None:
                decision = self._compute_target_status(frame, detection)
                self._draw_detection(frame, detection, decision)
                if self.debug:
                    print(f"Target detected: {detection['label']} | status={decision['status']} | dx={decision['dx']} dy={decision['dy']}")
            else:
                cv2.putText(frame, "Target not found", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                if self.debug:
                    print("Target not found")

            cv2.imshow("Drone Target Detection", frame)

            if self.writer is not None:
                self.writer.write(frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        if self.writer is not None:
            self.writer.release()
        cv2.destroyAllWindows()
