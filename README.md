# 📈 Smart Assembly Line  using Computer Vision

This project showcases how computer vision can be used to **monitor and improve the efficiency of assembly line operations** using video analysis and object detection.

By leveraging a YOLO-based deep learning model, the system provides **real-time insights** into station usage, operator behavior, and overall productivity—empowering manufacturers to make smarter, data-driven decisions.

---

<p align="center">
  <img src="https://github.com/user-attachments/assets/5d5f654e-d421-45b5-8201-800e10a679a7" alt="Assembly Line Monitoring Demo" width="600">
</p>

---

## 🚀 Features

- 🎥 Analyze assembly line video footage (live or recorded)
- 🧠 Detect and track workers, stations, and machines using a trained YOLO model
- 📊 Automatically determine operational status across multiple stations
- 🖼️ Overlay live status bars, timers, and ROI indicators on video output
- 💾 Save annotated videos for reporting and review

---

## 📁 Project Structure

```
.
├── main.py          # Core script to run monitoring on input videos
├── utils.py         # Helper functions (ROIs, overlays, productivity logic)
├── input/           # Folder to place input videos
├── output/          # Folder where output videos are saved
├── rois.json        # Stores selected Regions of Interest
├── runs/            # YOLO model training outputs (best.pt file inside)
└── README.md        # Project documentation
```

---

## 🧩 How It Works

1. 📹 Provide an input video of the assembly line.
2. 🔍 The system detects operators, stations, and machines using a trained YOLOv8 model.
3. 🧠 Logic interprets activity patterns to assess station performance.
4. 🖼️ Visual overlays (status bars, labels, timers) are drawn on the video.
5. 📤 The output is saved in the `output/` folder for review or reporting.

---

## ✅ Getting Started

### 🔧 Requirements

- Python 3.8+
- Ultralytics YOLOv8
- OpenCV
- NumPy

Install dependencies:

```bash
pip install -r requirements.txt
```

---

### ▶️ How to Run

1. Add your input video to the `input/` folder.
2. Make sure the trained model is saved at:
   ```
   runs/detect/train/weights/best.pt
   ```
3. Run the script:

```bash
python main.py
```

4. On first run, select the Regions of Interest (ROIs) for monitoring stations.
5. The processed output video will be saved to:  
   ```
   output/demo_out.mp4
   ```

---

## 🎯 Use Cases

- Monitor productivity and uptime of assembly line stations
- Detect worker-machine interaction and task completion
- Identify idle time, bottlenecks, or resource inefficiencies
- Generate annotated videos for analysis or operational audits

---

## 📄 License

This project is licensed under the **MIT License**.  
You're free to use, modify, and distribute it for personal or commercial purposes.

---

## 🙋‍♂️ Contact

Feel free to reach out for questions, feedback, or collaboration opportunities!
