import cv2
import numpy as np



# To make detections and get required outputs
def YOLO_Detection(model, frame, conf=0.9, iou = 0.3):
    # Perform inference on an image
    results = model.predict(frame, conf=conf, iou = iou)
    # Extract bounding boxes, classes, names, and confidences
    boxes = results[0].boxes.xyxy.tolist()
    classes = results[0].boxes.cls.tolist()
    names = results[0].names
    confidences = results[0].boxes.conf.tolist()

    return boxes, classes, names, confidences


## Draw YOLOv8 detections function
def label_detection(frame, text, left, top, bottom, right, tbox_color=(30, 155, 50), fontFace=1, fontScale=0.8,
                    fontThickness=1):
    # Draw Bounding Box
    cv2.rectangle(frame, (int(left), int(top)), (int(bottom), int(right)), tbox_color, 1)
    # Draw and Label Text
    textSize = cv2.getTextSize(text, fontFace, fontScale, fontThickness)
    text_w = textSize[0][0]
    text_h = textSize[0][1]
    y_adjust = 10
    cv2.rectangle(frame, (int(left), int(top) + text_h + y_adjust), (int(left) + text_w + y_adjust, int(top)),
                  tbox_color, -1)
    cv2.putText(frame, text, (int(left) + 5, int(top) + 10), fontFace, fontScale, (255, 255, 255), fontThickness,
                cv2.LINE_AA)


def draw_status_overlay(frame, statuses):
    """
    Draws transparent status rectangles and indicators for each variable.
    :param statuses: list of tuples [(label, status_bool), ...]
    """
    overlay = frame.copy()
    h, w, _ = frame.shape

    # Settings for a smaller video frame
    start_x, start_y = 10, 10
    box_w, box_h = 180, 50
    spacing = 10

    font_scale = 0.5
    circle_radius = 10
    text_color = (255, 255, 255)

    for i, (label, status) in enumerate(statuses):
        x1 = start_x
        y1 = start_y + i * (box_h + spacing)
        x2 = x1 + box_w
        y2 = y1 + box_h

        # Draw semi-transparent background
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (50, 50, 50), -1)

        # Status circle
        circle_color = (0, 255, 0) if status else (0, 0, 255)
        cv2.circle(overlay, (x1 + 20, y1 + box_h // 2), circle_radius, circle_color, -1)

        # Status label
        cv2.putText(overlay, label, (x1 + 40, y1 + box_h // 2 + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 1)

    # Blend with the original frame
    frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)
    return frame


def select_multiple_rois(frame):
    print("Select 3 ROIs. Press ENTER after each, ESC to finish.")
    rois = cv2.selectROIs("Select ROIs", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select ROIs")

    if len(rois) != 3:
        print("You must select exactly 3 ROIs.")
        return None

    # Convert to (x1, y1, x2, y2) format
    roi_coords = []
    for roi in rois:
        x, y, w, h = roi
        roi_coords.append([(x, y), (x + w, y + h)])
    return roi_coords


def pad_frame_with_status(frame, statuses, productivity_times):
    h, w = frame.shape[:2]
    banner_height = 140  # increased to fit extra text
    new_frame = np.full((h + banner_height, w, 3), 255, dtype=np.uint8)
    new_frame[banner_height:, :] = frame

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    thickness = 1

    station_names = ["Station A", "Station B", "Station C"]

    for i, ((person_status, station_status), prod_time) in enumerate(zip(statuses, productivity_times)):
        section_width = w // len(statuses)
        x_start = i * section_width + 20  # Slightly increased X
        y_start = 25

        # Station name
        cv2.putText(new_frame, station_names[i], (x_start + 25, y_start), font, 0.55, (0, 0, 0), 2)

        # Operator status circle & text
        person_color = (0, 200, 0) if person_status == "working" else (0, 0, 255)
        cv2.circle(new_frame, (x_start + 5, y_start + 20), 6, person_color, -1)
        cv2.putText(new_frame, f"Operator: {person_status.upper()}",
                    (x_start + 20, y_start + 25), font, font_scale, (0, 0, 0), thickness)

        # Workstation status circle & text
        station_color = (0, 200, 0) if station_status in ["on_station", "engine_locked"] else (0, 0, 255)
        station_label = "ON" if station_status in ["on_station", "engine_locked"] else "off"
        cv2.circle(new_frame, (x_start + 5, y_start + 45), 6, station_color, -1)
        cv2.putText(new_frame, f"Workstation: {station_label.upper()}",
                    (x_start + 20, y_start + 50), font, font_scale, (0, 0, 0), thickness)

        # Productivity time (in seconds) - bold effect by increasing thickness
        if i == 0:
            prod_text = f"Productivity: {round(prod_time, 2)} min"
            cv2.putText(new_frame, prod_text,
                        (x_start + 20, y_start + 100), font, font_scale - 0.05, (0, 0, 0), 1)  # thickness=3 makes text bold
        elif i == 1:
            prod_text = f"Productivity: {round(prod_time, 2)} min"
            cv2.putText(new_frame, prod_text,
                        (x_start + 40, y_start + 100), font, font_scale - 0.05, (0, 0, 0), 1)  # thickness=3 makes text bold
        elif i == 2:
            prod_text = f" Productivity: {round(prod_time, 2)} min"
            cv2.putText(new_frame, prod_text,
                        (x_start + 25, y_start + 100), font, font_scale - 0.05, (0, 0, 0), 1)  # thickness=3 makes text bold

    return new_frame


def draw_transparent_status_bar(frame, statuses):
    """Draw a small transparent white status bar at the top with status indicators."""
    overlay = frame.copy()
    bar_height = 40
    bar_color = (255, 255, 255)  # White
    alpha = 0.6

    # Draw transparent white bar
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], bar_height), bar_color, thickness=-1)
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    # Draw status texts
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 1
    x = 10

    for label, is_on in statuses:
        color = (0, 200, 0) if is_on else (0, 0, 255)
        cv2.circle(frame, (x + 10, 20), 6, color, -1)
        cv2.putText(frame, label, (x + 25, 25), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
        x += 160  # Spacing between status blocks

    return frame


def select_roi(frame):
    """
    Lets the user select an ROI rectangle on the frame using OpenCV GUI.
    Returns (x1, y1, x2, y2)
    """
    roi = cv2.selectROI("Select ROI (press ENTER or SPACE to confirm, ESC to cancel)", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select ROI (press ENTER or SPACE to confirm, ESC to cancel)")

    if roi == (0, 0, 0, 0):
        return None  # User canceled
    x, y, w, h = roi
    return (x, y, x + w, y + h)

def is_in_roi(bbox, roi):
    (rx1, ry1), (rx2, ry2) = roi  # <-- Unpacks two tuples
    x1, y1, x2, y2 = bbox

    return rx1 <= x1 and ry1 <= y1 and x2 <= rx2 and y2 <= ry2

def draw_rois(frame, rois):
    for i, roi in enumerate(rois):
        top_left, bottom_right = roi
        cv2.rectangle(frame, top_left, bottom_right, (50, 50, 200), thickness=1)  # Thin outline

def is_center_in_roi(box, roi):
    x1, y1, x2, y2 = map(int, box)
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    (rx1, ry1), (rx2, ry2) = roi
    return rx1 <= cx <= rx2 and ry1 <= cy <= ry2
# def setup_device():
#     """Check if CUDA is available and set the device."""
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print(f"Using device: {device}")
#     return device

