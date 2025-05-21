import os
import json
import time
import cv2
from ultralytics import YOLO
from utilis import (
    YOLO_Detection, label_detection, select_roi, is_in_roi,
    draw_status_overlay, draw_transparent_status_bar,
    pad_frame_with_status, select_multiple_rois, is_center_in_roi
)


def load_yolo_model(device, classes):
    model = YOLO("runs/detect/train/weights/best.pt", classes)
    model.to(device)
    model.nms = 0.8
    print(f"Model classes: {model.names}")
    return model

def convert_to_int(roi_list):
    return [[[int(x) for x in pt1], [int(x) for x in pt2]] for pt1, pt2 in roi_list]

station_states = ['off', 'off', 'off']
productivity_times = [0.00, 11.34, 21.21]  # in seconds
# Add this global outside functions
last_update_time = time.time()
working_flags = [False, False, False]
working_start_times = [None, None, None]

def process_frame(model, frame, rois):
    global station_states, productivity_times, last_update_time

    current_time = time.time()
    dt = current_time - last_update_time
    last_update_time = current_time

    boxes, classes, names, confidences = YOLO_Detection(model, frame, conf=0.6)

    person_statuses = ["None"] * len(rois)
    station_flags = [{"on_station": False, "engine": False} for _ in rois]

    for box, cls in zip(boxes, classes):
        label = names[int(cls)]
        x1, y1, x2, y2 = map(int, box)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if label == "working":
            color = (50, 205, 50)  # Green
        elif label == "waiting":
            color = (0, 0, 255)  # Red
        else:
            color = (255, 144, 30)  # Orange

        for i, roi in enumerate(rois):
            (rx1, ry1), (rx2, ry2) = roi
            if label in ["working", "waiting"] and is_center_in_roi(box, roi):
                person_statuses[i] = label
            elif label == "on_station" and is_center_in_roi(box, roi):
                station_flags[i]["on_station"] = True
            elif label == "engine" and is_center_in_roi(box, roi):
                station_flags[i]["engine"] = True

        label_detection(frame=frame, text=label, tbox_color=color,
                        left=box[0], top=box[1], bottom=box[2], right=box[3])

    for i in range(len(rois)):
        if station_flags[i]["engine"]:
            station_states[i] = "engine_locked"
        elif not station_flags[i]["on_station"] and not station_flags[i]["engine"]:
            station_states[i] = "off"
        elif station_flags[i]["on_station"] and station_states[i] != "engine_locked":
            station_states[i] = "on_station"

        # Update productivity time only if operator is working
        if person_statuses[i] == "working":
            productivity_times[i] += dt / 60

    # Draw ROI rectangles
    for i, roi in enumerate(rois):
        (x1, y1), (x2, y2) = roi
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 1)

    # Add top status banner with productivity times
    statuses = list(zip(person_statuses, station_states))
    frame_with_status = pad_frame_with_status(frame, statuses, productivity_times)
    return frame_with_status


def main(source, output_path="output_video/demo_out.mp4"):
    model = load_yolo_model(device="cuda", classes=[0, 56, 60])

    if os.path.isfile(source) and not source.lower().endswith(('.png', '.jpg')):
        cap = cv2.VideoCapture(source)
        ret, first_frame = cap.read()
        if not ret:
            print("Failed to read video")
            return

        if os.path.exists("rois.json"):
            with open("rois.json", "r") as f:
                rois = json.load(f)
            rois = [tuple(map(tuple, r)) for r in rois]
            print("Loaded saved ROIs:", rois)
        else:
            rois = select_multiple_rois(first_frame)
            with open("rois.json", "w") as f:
                json.dump(convert_to_int(rois), f)
            print("Saved selected ROIs:", rois)

        # Make sure output folder exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Prepare VideoWriter with original FPS
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        frame_height, frame_width = first_frame.shape[:2]

        fps = 15

        banner_height = 110  # must match pad_frame_with_status banner height

        out = cv2.VideoWriter(output_path, fourcc, 15,
                              (frame_width, 476))

        if not out.isOpened():
            print("Failed to open VideoWriter")
            return

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = process_frame(model, frame, rois)
            print(frame.shape)
            out.write(frame)  # Write processed frame to video

            cv2.imshow('Frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        out.release()
        print(f"Output saved to {output_path}")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main(source="input/engine_1.mp4", output_path="output/demo_out.mp4")
