# Object-Detection
# 👁️ VisionAI - Object Detection

A modern AI-powered Object Detection application built with **Python, Streamlit, and YOLOv8**.

VisionAI can detect objects in **images, videos, and webcam captures** and provides a visual detection result along with object counts and a summary.

---

## ✨ Features

- 🖼️ Object detection in images
- 🎥 Object detection in videos
- 📹 Webcam image detection
- 🎯 Adjustable confidence threshold
- 📊 Object counting and detection statistics
- 🏷️ Detection of multiple object classes
- ⬇️ Download processed videos
- ⚡ Fast inference using YOLOv8 Nano
- 🎨 Modern dark UI built with Streamlit
- 📈 Progress bar for video processing

---

## 🤖 Technologies

- **Python**
- **Streamlit**
- **YOLOv8**
- **Ultralytics**
- **OpenCV**
- **Pillow**
- **NumPy**

---

## 🧠 How It Works

The application uses the **YOLOv8 Nano** model from Ultralytics to detect objects.

The workflow is:

```text
Input
  │
  ├── Image
  ├── Video
  └── Webcam
       │
       ▼
   YOLOv8 Model
       │
       ▼
 Object Detection
       │
       ├── Bounding Boxes
       ├── Object Classes
       └── Object Counts
       │
       ▼
 Detection Results
