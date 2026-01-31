"""
Run YOLOv5 detection with Flask streaming, severity classification, and AWS SNS notifications.

Custom Logic:
- High Severity: Fire, people, and other classes detected.
- Medium Severity: (Fire and people) or (fire and other classes, but not people).
- Low Severity: Only fire detected.
- Annotate severity on frame and send AWS SNS notifications.

Usage:
    $ python detect.py --weights yolov5s.pt --source 0 --save-csv --sns-topic-arn <your-sns-topic-arn>
"""

import argparse
import csv
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from threading import Lock

import boto3
import cv2
import torch
from flask import Flask, Response

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

from ultralytics.utils.plotting import Annotator, colors, save_one_box
from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (
    LOGGER,
    Profile,
    check_file,
    check_img_size,
    check_imshow,
    check_requirements,
    colorstr,
    cv2,
    increment_path,
    non_max_suppression,
    print_args,
    scale_boxes,
    strip_optimizer,
    xyxy2xywh,
)
from utils.torch_utils import select_device, smart_inference_mode

# Initialize Flask app
app = Flask(__name__)

# CSV lock for thread-safe writing
csv_lock = Lock()

# AWS SNS client
sns_client = None


def setup_logging(verbose=False):
    """Configure logging with custom format and verbosity."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def initialize_sns(sns_topic_arn):
    """Initialize AWS SNS client."""
    global sns_client
    try:
        sns_client = boto3.client("sns")
        # Verify topic exists
        sns_client.get_topic_attributes(TopicArn=sns_topic_arn)
        LOGGER.info(f"AWS SNS initialized with topic: {sns_topic_arn}")
    except Exception as e:
        LOGGER.error(f"Failed to initialize AWS SNS: {e}")
        sns_client = None


def send_sns_notification(sns_topic_arn, severity, timestamp):
    """Send notification via AWS SNS."""
    if sns_client is None:
        LOGGER.warning("SNS client not initialized, skipping notification")
        return
    try:
        message = f"Severity Alert: {severity}\nTimestamp: {timestamp}\nDetected in video stream."
        sns_client.publish(
            TopicArn=sns_topic_arn,
            Message=message,
            Subject=f"YOLOv5 Severity {severity} Alert",
        )
        LOGGER.info(f"Sent SNS notification: {severity} at {timestamp}")
    except Exception as e:
        LOGGER.error(f"Failed to send SNS notification: {e}")


def determine_severity(detected_classes, names):
    """Determine severity based on detected classes."""
    has_fire = "fire" in detected_classes
    has_person = "person" in detected_classes
    has_other = any(cls not in ["fire", "person"] for cls in detected_classes)

    if has_fire and has_person and has_other:
        return "High"
    elif (has_fire and has_person) or (has_fire and has_other and not has_person):
        return "Medium"
    elif has_fire and not has_person and not has_other:
        return "Low"
    return None


@smart_inference_mode()
def run(
    weights=ROOT / "yolov5s.pt",
    source=ROOT / "data/images",
    data=ROOT / "data/coco128.yaml",
    imgsz=(640, 640),
    conf_thres=0.25,
    iou_thres=0.45,
    max_det=1000,
    device="",
    view_img=False,
    save_txt=False,
    save_csv=False,
    save_conf=False,
    save_crop=False,
    nosave=False,
    classes=None,
    agnostic_nms=False,
    augment=False,
    visualize=False,
    update=False,
    project=ROOT / "runs/detect",
    name="exp",
    exist_ok=False,
    line_thickness=3,
    hide_labels=False,
    hide_conf=False,
    half=False,
    dnn=False,
    vid_stride=1,
    verbose=False,
    sns_topic_arn=None,
):
    """Run YOLOv5 detection with Flask streaming, severity classification, and SNS notifications."""
    source = str(source)
    save_img = not nosave and not source.endswith(".txt")
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
    webcam = source.isnumeric() or source.endswith(".streams") or (is_url and not is_file)
    screenshot = source.lower().startswith("screen")
    if is_url and is_file:
        source = check_file(source)

    # Initialize SNS if topic ARN provided
    if sns_topic_arn:
        initialize_sns(sns_topic_arn)

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)
    (save_dir / "labels" if save_txt else save_dir).mkdir(parents=True, exist_ok=True)

    # Load model
    try:
        device = select_device(device)
        model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
        stride, names, pt = model.stride, model.names, model.pt
        imgsz = check_img_size(imgsz, s=stride)
    except Exception as e:
        LOGGER.error(f"Failed to load model: {e}")
        return

    # Verify required classes
    required_classes = ["fire", "person"]
    if not all(cls in names for cls in required_classes):
        LOGGER.error(f"Model must support classes: {required_classes}. Found: {names}")
        return

    # Dataloader
    bs = 1
    try:
        if webcam:
            view_img = check_imshow(warn=True)
            dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
            bs = len(dataset)
        elif screenshot:
            dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
        else:
            dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    except Exception as e:
        LOGGER.error(f"Failed to load data source: {e}")
        return

    # Initialize class counts
    class_counts = {}

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))
    seen, dt = 0, (Profile(device=device), Profile(device=device), Profile(device=device))
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()
            im /= 255
            if len(im.shape) == 3:
                im = im[None]
            if model.xml and im.shape[0] > 1:
                ims = torch.chunk(im, im.shape[0], 0)

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            if model.xml and im.shape[0] > 1:
                pred = None
                for image in ims:
                    if pred is None:
                        pred = model(image, augment=augment, visualize=visualize).unsqueeze(0)
                    else:
                        pred = torch.cat((pred, model(image, augment=augment, visualize=visualize).unsqueeze(0)), dim=0)
                pred = [pred, None]
            else:
                pred = model(im, augment=augment, visualize=visualize)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # CSV path
        csv_path = save_dir / "predictions.csv"

        # Write to CSV
        def write_to_csv(image_name, prediction, confidence, xyxy, severity):
            with csv_lock:
                data = {
                    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Image Name": image_name,
                    "Prediction": prediction,
                    "Confidence": confidence,
                    "Bounding Box": f"[{xyxy[0]:.1f}, {xyxy[1]:.1f}, {xyxy[2]:.1f}, {xyxy[3]:.1f}]",
                    "Severity": severity or "None",
                }
                file_exists = os.path.isfile(csv_path)
                with open(csv_path, mode="a", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=data.keys())
                    if not file_exists:
                        writer.writeheader()
                    writer.writerow(data)

        # Process predictions
        for i, det in enumerate(pred):
            seen += 1
            if webcam:
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f"{i}: "
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, "frame", 0)

            p = Path(p)
            save_path = str(save_dir / p.name)
            txt_path = str(save_dir / "labels" / p.stem) + ("" if dataset.mode == "image" else f"_{frame}")
            s += "{:g}x{:g} ".format(*im.shape[2:])
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
            imc = im0.copy() if save_crop else im0
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))

            # Track detected classes for severity
            detected_classes = set()

            if len(det):
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Log detection summary
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "
                    detected_classes.add(names[int(c)])

                # Process detections
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)
                    label = names[c] if hide_conf else f"{names[c]}"
                    confidence = float(conf)
                    confidence_str = f"{confidence:.2f}"

                    # Update class counts
                    class_counts[label] = class_counts.get(label, 0) + 1

                    # Log to CLI
                    LOGGER.info(
                        f"Detection: {label}, Confidence: {confidence_str}, "
                        f"Box: [{xyxy[0]:.1f}, {xyxy[1]:.1f}, {xyxy[2]:.1f}, {xyxy[3]:.1f}]"
                    )

                    # Annotate image
                    if save_img or save_crop or view_img:
                        label = None if hide_labels else (names[c] if hide_conf else f"{names[c]} {conf:.2f}")
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / "crops" / names[c] / f"{p.stem}.jpg", BGR=True)

            # Determine severity
            severity = determine_severity(detected_classes, names)
            if severity:
                LOGGER.info(f"Severity Detected: {severity}")
                # Annotate severity on frame
                severity_text = f"Severity: {severity}"
                severity_color = (
                    (0, 0, 255) if severity == "High" else (0, 165, 255) if severity == "Medium" else (0, 255, 255)
                )
                cv2.putText(
                    im0,
                    severity_text,
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    severity_color,
                    2,
                    cv2.LINE_AA,
                )
                # Send SNS notification
                if sns_topic_arn:
                    send_sns_notification(sns_topic_arn, severity, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

            # Save to CSV and txt
            if len(det):
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)
                    label = names[c] if hide_conf else f"{names[c]}"
                    confidence = float(conf)
                    confidence_str = f"{confidence:.2f}"
                    if save_csv:
                        write_to_csv(p.name, label, confidence_str, xyxy, severity)
                    if save_txt:
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)
                        with open(f"{txt_path}.txt", "a") as f:
                            f.write(("%g " * len(line)).rstrip() % line + "\n")

            # Add class counts to frame
            count_text = ", ".join([f"{count} {label}" for label, count in class_counts.items()])
            cv2.putText(im0, count_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2, cv2.LINE_AA)

            # Stream frame
            im0 = annotator.result()
            ret, buffer = cv2.imencode(".jpg", im0, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if not ret:
                LOGGER.error("Failed to encode frame")
                continue
            frame = buffer.tobytes()
            yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

        # Log performance
        if verbose:
            LOGGER.debug(f"{s}{'no detections' if not len(det) else ''}, Inference: {dt[1].dt * 1e3:.1f}ms")

    # Final performance metrics
    t = tuple(x.t / seen * 1e3 for x in dt)
    LOGGER.info(f"Speed: {t[0]:.1f}ms pre-process, {t[1]:.1f}ms inference, {t[2]:.1f}ms NMS per image at shape {(1, 3, *imgsz)}")
    if save_txt or save_img or save_csv:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ""
        s += f"\nResults saved to CSV at {csv_path}" if save_csv else ""
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])


@app.route("/")
def index():
    """Stream video feed with detections."""
    try:
        return Response(run(**vars(opt)), mimetype="multipart/x-mixed-replace; boundary=frame")
    except Exception as e:
        LOGGER.error(f"Streaming error: {e}")
        return "Streaming error", 500


def parse_opt():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", nargs="+", type=str, default=ROOT / "yolov5s.pt", help="model path or triton URL")
    parser.add_argument("--source", type=str, default=ROOT / "data/images", help="file/dir/URL/glob/screen/0(webcam)")
    parser.add_argument("--data", type=str, default=ROOT / "data/coco128.yaml", help="(optional) dataset.yaml path")
    parser.add_argument("--imgsz", "--img", "--img-size", nargs="+", type=int, default=[640], help="inference size h,w")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=1000, help="maximum detections per image")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--view-img", action="store_true", help="show results")
    parser.add_argument("--save-txt", action="store_true", help="save results to *.txt")
    parser.add_argument("--save-csv", action="store_true", help="save results in CSV format")
    parser.add_argument("--save-conf", action="store_true", help="save confidences in --save-txt labels")
    parser.add_argument("--save-crop", action="store_true", help="save cropped prediction boxes")
    parser.add_argument("--nosave", action="store_true", help="do not save images/videos")
    parser.add_argument("--classes", nargs="+", type=int, help="filter by class: --classes 0, or --classes 0 2 3")
    parser.add_argument("--agnostic-nms", action="store_true", help="class-agnostic NMS")
    parser.add_argument("--augment", action="store_true", help="augmented inference")
    parser.add_argument("--visualize", action="store_true", help="visualize features")
    parser.add_argument("--update", action="store_true", help="update all models")
    parser.add_argument("--project", default=ROOT / "runs/detect", help="save results to project/name")
    parser.add_argument("--name", default="exp", help="save results to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--line-thickness", default=3, type=int, help="bounding box thickness (pixels)")
    parser.add_argument("--hide-labels", default=False, action="store_true", help="hide labels")
    parser.add_argument("--hide-conf", default=False, action="store_true", help="hide confidences")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument("--dnn", action="store_true", help="use OpenCV DNN for ONNX inference")
    parser.add_argument("--vid-stride", type=int, default=1, help="video frame-rate stride")
    parser.add_argument("--verbose", action="store_true", help="verbose logging")
    parser.add_argument("--sns-topic-arn", type=str, default=None, help="AWS SNS topic ARN for notifications")
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1
    print_args(vars(opt))
    return opt


def main(opt):
    """Main execution function."""
    setup_logging(opt.verbose)
    check_requirements(ROOT / "requirements.txt", exclude=("tensorboard", "thop"))
    LOGGER.info("Starting Flask server on http://0.0.0.0:5050")
    app.run(host="0.0.0.0", port=5050, debug=False)


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)