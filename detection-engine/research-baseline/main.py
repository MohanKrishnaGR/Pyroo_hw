import cv2
from flask import Flask, Response
from pathlib import Path
from utils.general import non_max_suppression, scale_boxes, xyxy2xywh, increment_path
from utils.plots import colors
import argparse
import csv
import os
import sys
import time
import logging
import boto3
import torch
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
    increment_path,
    non_max_suppression,
    print_args,
    scale_boxes,
    strip_optimizer,
    xyxy2xywh
)
from utils.torch_utils import select_device, smart_inference_mode


# Initialize Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('detection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
from twilio.rest import Client
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(dotenv_path=ROOT.parent / ".env")

TWILIO_ACCOUNT_SID = os.getenv('TWILIO_ACCOUNT_SID')
TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN')
TWILIO_FROM = os.getenv('TWILIO_FROM')
TWILIO_TO = os.getenv('TWILIO_TO')

# Initialize Twilio client only if credentials are provided
twilio_client = None
if TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN:
    twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
else:
    logger.warning("Twilio credentials not found in environment variables. SMS notifications disabled.")

NOTIFICATION_INTERVAL = 5  # Seconds between notifications

# AWS SNS setup
# Load SNS configuration from .env if needed
# sns_client = boto3.client('sns', region_name=os.getenv('AWS_REGION', 'ap-northeast-1'))
# SNS_TOPIC_ARN = os.getenv('SNS_TOPIC_ARN')
# NOTIFICATION_INTERVAL = 30  # Seconds between notifications

@smart_inference_mode()
def generate(
        weights=ROOT / 'best_weight.engine',
        source=ROOT / '0',  # Default to webcam index 0
        data=ROOT / 'data.yaml',
        imgsz=(640, 640),
        conf_thres=0.25,
        iou_thres=0.45,
        max_det=1000,
        device='',
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
        project=ROOT / 'runs/detect',
        name='exp',
        exist_ok=False,
        line_thickness=3,
        hide_labels=False,
        hide_conf=False,
        half=False,
        dnn=False,
        vid_stride=1,
):
    source = str(source)
    save_img = not nosave and not source.endswith(".txt")
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
    webcam = source.isnumeric() or source.endswith(".streams") or (is_url and not is_file)
    screenshot = source.lower().startswith("screen")
    if is_url and is_file:
        source = check_file(source)

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)
    (save_dir / "labels" if save_txt else save_dir).mkdir(parents=True, exist_ok=True)

    # Load model
    device = select_device(device)  # Corrected device selection
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)

    # Dataloader
    bs = 1
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    last_notification_time = time.time()

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()
            im /= 255
            if len(im.shape) == 3:
                im = im[None]
            if model.xml and im.shape[0] > 1:
                ims = torch.chunk(im, im.shape[0], 0)

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

        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # CSV path
        csv_path = save_dir / "predictions.csv"

        def write_to_csv(image_name, prediction, confidence):
            data = {"Image Name": image_name, "Prediction": prediction, "Confidence": confidence}
            with open(csv_path, mode="a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=data.keys())
                if not csv_path.is_file():
                    writer.writeheader()
                writer.writerow(data)

        class_counts = {}
        severity = None
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
            s += "%gx%g " % im.shape[2:]
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
            imc = im0.copy() if save_crop else im0
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))

            detected_classes = set()
            if len(det):
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "
                    detected_classes.add(names[int(c)])
                    class_counts[names[int(c)]] = class_counts.get(names[int(c)], 0) + n

                # Severity detection
                other_classes = detected_classes - {'fire', 'people'}
                if 'fire' in detected_classes:
                    if 'people' in detected_classes and other_classes:
                        severity = 'High'
                    elif 'people' in detected_classes or other_classes:
                        severity = 'Medium'
                    else:
                        severity = 'Low'

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)
                    label = names[c] if hide_conf else f"{names[c]}"
                    confidence = float(conf)
                    confidence_str = f"{confidence:.2f}"
                    label_text = f"{label} {confidence_str}"
                    cv2.putText(im0, label_text, (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
                    cv2.rectangle(im0, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)

                    if save_csv:
                        write_to_csv(p.name, label, confidence_str)

                    if save_txt:
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)
                        with open(f"{txt_path}.txt", "a") as f:
                            f.write(("%g " * len(line)).rstrip() % line + "\n")

                    if save_img or save_crop or view_img:
                        c = int(cls)
                        label = None if hide_labels else (names[c] if hide_conf else f"{names[c]} {conf:.2f}")
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / "crops" / names[c] / f"{p.stem}.jpg", BGR=True)

            # Annotate severity on frame
            if severity:
                severity_text = f"Severity: {severity}"
                severity_color = (0, 0, 255) if severity == 'High' else (0, 165, 255) if severity == 'Medium' else (0, 255, 255)
                cv2.putText(im0, severity_text, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, severity_color, 2, cv2.LINE_AA)
                current_time = time.time()
                if twilio_client and (current_time - last_notification_time > NOTIFICATION_INTERVAL):
                    try:
                        message = twilio_client.messages.create(
                            # body=f"🔥 PyroGuardian Alert: {severity} severity detected at {time.strftime('%Y-%m-%d %H:%M:%S')} with classes {list(detected_classes)}"
                            body = "🔥 PyroGuardian Alert: {} severity detected at {} with classes {}".format(
                                severity, time.strftime('%Y-%m-%d %H:%M:%S'), list(detected_classes)
                            ),
                            # body=f"[PyroGuardian] Alert! {severity} severity detected in video feed."
                            from_=TWILIO_FROM,
                            to=TWILIO_TO
                        )
                        logger.info(f"SMS sent successfully: {message.sid}")
                        last_notification_time = current_time
                    except Exception as e:
                        logger.error(f"Failed to send SMS: {e}")


            # Display class counts
            count_text = ", ".join([f"{count} {label}" for label, count in class_counts.items()])
            cv2.putText(im0, count_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2, cv2.LINE_AA)

            # Log frame details
            log_message = f"Frame {seen}: Detected {class_counts}, Severity: {severity if severity else 'None'}"
            logger.info(log_message)

            # Send SNS notification if needed
            # if severity and (time.time() - last_notification_time) >= NOTIFICATION_INTERVAL:
            #     message = f"Alert: Severity {severity} detected at {time.strftime('%Y-%m-%d %H:%M:%S')}. Classes: {class_counts}"
            #     try:
            #         sns_client.publish(
            #             TopicArn=SNS_TOPIC_ARN,
            #             Message=message,
            #             Subject=f"Safety Alert: {severity} Severity"
            #         )
            #         logger.info(f"SNS notification sent: {message}")
            #         last_notification_time = time.time()
            #     except Exception as e:
            #         logger.error(f"Failed to send SNS notification: {str(e)}")
            
            # Send Twilio notification if needed
            # if severity and (time.time() - last_notification_time) >= NOTIFICATION_INTERVAL:
            #     try:
            #         message_body = f"🔥 Alert: {severity} severity detected at {time.strftime('%Y-%m-%d %H:%M:%S')} with classes {list(detected_classes)}"
            #         message = twilio_client.messages.create(
            #             body=message_body,
            #             from_=TWILIO_FROM,
            #             to=TWILIO_TO
            #         )
            #         logger.info(f"Twilio alert sent. SID: {message.sid}")
            #         last_notification_time = time.time()
            #     except Exception as e:
            #         logger.error(f"Error sending Twilio SMS: {e}")


            im0 = annotator.result()
            ret, buffer = cv2.imencode('.jpg', im0)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return Response(generate(**vars(opt)), mimetype='multipart/x-mixed-replace; boundary=frame')

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", nargs="+", type=str, default=ROOT / "yolov5s.pt", help="model path or triton URL")
    parser.add_argument("--source", type=str, default='0', help="webcam index (0), file/dir/URL/glob/screen")  # Corrected default
    parser.add_argument("--data", type=str, default=ROOT / "data/coco128.yaml", help="dataset.yaml path")
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
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1
    print_args(vars(opt))
    return opt

if __name__ == '__main__':
    opt = parse_opt()
    check_requirements(ROOT / "requirements.txt", exclude=("tensorboard", "thop"))
    app.run(host='0.0.0.0', port=5050, debug=True)