#!/usr/bin/env python3

import sys
import os
import time
from datetime import datetime
import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib
import pyds  # NVIDIA DeepStream bindings

# Import our notification utility
from core.notifications import NotificationManager, determine_severity

# PGIE (Primary GPU Inference Engine) Class IDs based on RT-DETR/YOLO mapping
PGIE_CLASS_ID_FIRE = 0
PGIE_CLASS_ID_PERSON = 1


def osd_sink_pad_buffer_probe(pad, info, u_data):
    """
    Probe function to access metadata from the buffer.
    This is where we implement severity logic and trigger alerts.
    """
    frame_number = 0
    num_rects = 0
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to get GstBuffer ")
        return Gst.PadProbeReturn.OK

    # Retrieve batch metadata from the gst_buffer
    batch_meta = pyds.gst_buffer_get_nvbatch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        frame_number = frame_meta.frame_num
        num_rects = frame_meta.num_obj_meta
        l_obj = frame_meta.obj_meta_list

        detected_classes = set()

        while l_obj is not None:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break

            # Identify classes (Assuming mapping: 0=fire, 1=person, etc.)
            if obj_meta.class_id == PGIE_CLASS_ID_FIRE:
                detected_classes.add("fire")
            elif obj_meta.class_id == PGIE_CLASS_ID_PERSON:
                detected_classes.add("person")
            else:
                detected_classes.add("other")

            try:
                l_obj = l_obj.next
            except StopIteration:
                break

        # Check severity and notify
        severity = determine_severity(detected_classes)
        if severity and (frame_number % 30 == 0):  # Notify once per second at 30fps
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"Frame {frame_number}: {severity} severity fire detected!")
            u_data.send_sns_notification(severity, timestamp)

        try:
            l_frame = l_frame.next
        except StopIteration:
            break
    return Gst.PadProbeReturn.OK


def main(args):
    # Initialize Notification Manager
    sns_arn = args[1] if len(args) > 1 else None
    notifier = NotificationManager(sns_topic_arn=sns_arn)

    # Standard GStreamer initialization
    Gst.init(None)

    # Create Pipeline
    pipeline = Gst.Pipeline()

    # Create Elements
    source = Gst.ElementFactory.make("nvv4l2camerasrc", "camera-source")
    caps = Gst.ElementFactory.make("capsfilter", "filter")
    caps.set_property(
        "caps",
        Gst.Caps.from_string(
            "video/x-raw(memory:NVMM), width=1280, height=720, format=UYVY, framerate=30/1"
        ),
    )

    vidconv = Gst.ElementFactory.make("nvvideoconvert", "converter")
    nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
    sink = Gst.ElementFactory.make("nveglglessink", "display-sink")

    # The Inference Engine (RT-DETR via TensorRT)
    pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
    pgie.set_property("config-file-path", "rtdetr_config.txt")

    # Add elements to pipeline
    pipeline.add(source)
    pipeline.add(caps)
    pipeline.add(pgie)
    pipeline.add(vidconv)
    pipeline.add(nvosd)
    pipeline.add(sink)

    # Link elements
    source.link(caps)
    caps.link(pgie)
    pgie.link(vidconv)
    vidconv.link(nvosd)
    nvosd.link(sink)

    # Add probe to OSD sink pad to run our custom logic
    osdsinkpad = nvosd.get_static_pad("sink")
    if not osdsinkpad:
        sys.stderr.write(" Unable to get sink pad of nvosd \n")
    else:
        osdsinkpad.add_probe(
            Gst.PadProbeType.BUFFER, osd_sink_pad_buffer_probe, notifier
        )

    # Start playing
    loop = GLib.MainLoop()
    pipeline.set_state(Gst.State.PLAYING)

    try:
        loop.run()
    except:
        pass

    # Clean up
    pipeline.set_state(Gst.State.NULL)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
