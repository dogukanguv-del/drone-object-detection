import argparse
import sys
from pathlib import Path

from src.detector import DroneTargetDetector


def parse_args():
    parser = argparse.ArgumentParser(description="Drone target detection using OpenCV and YOLO")
    parser.add_argument("--source", type=str, default="0", help="Camera index or video file path")
    parser.add_argument("--mode", type=str, default="color", choices=["color", "circle", "yolo"],
                        help="Detection mode: color, circle, or yolo")
    parser.add_argument("--color", type=str, default="red", choices=["red", "green", "blue", "yellow", "orange", "purple"],
                        help="Color target used in color mode")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="YOLO model path")
    parser.add_argument("--conf", type=float, default=0.5, help="YOLO confidence threshold")
    parser.add_argument("--output", type=str, default=None, help="Optional output video file")
    parser.add_argument("--show-debug", action="store_true", help="Display additional debug information")
    return parser.parse_args()


def main():
    args = parse_args()

    detector = DroneTargetDetector(
        source=args.source,
        mode=args.mode,
        target_color=args.color,
        model_path=args.model,
        conf_threshold=args.conf,
        output_path=args.output,
        debug=args.show_debug,
    )

    try:
        detector.run()
    except KeyboardInterrupt:
        print("\nProgram interrupted by user.")
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
