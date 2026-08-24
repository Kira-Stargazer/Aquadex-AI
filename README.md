# 🌊 Aquadex AI

## AI-Powered Automated Underwater Marine Debris and Anomaly Detection Using Side-Scan Sonar Imagery

Aquadex AI is an AI-assisted research prototype designed to detect potential man-made anomalies in side-scan sonar imagery.

The system uses YOLO11n-Seg for object detection and segmentation and provides confidence scoring, localization information, annotated imagery, and structured JSON/CSV reporting through a Streamlit dashboard.

---

# 🎯 Objective

Side-scan sonar imagery contains challenging acoustic patterns such as speckle noise, acoustic shadows, natural seafloor structures, and varying image characteristics.

Aquadex AI aims to assist marine researchers by automatically identifying candidate man-made anomalies in side-scan sonar imagery.

---

# 🚀 Features

- Side-scan sonar image upload
- YOLO11n-Seg detection
- Pixel-level segmentation
- Confidence scoring
- Candidate anomaly filtering
- GPS/location information
- Google Maps location link
- Annotated sonar imagery
- JSON report generation
- CSV report generation
- Streamlit dashboard
- Lightweight YOLO architecture

---

# 🌊 AquaDex AI

An AI-powered application built with Streamlit.

# 🌐 Try the App

👉[Launch AquaDex AI](https://aquadex-ai-nasuxwargfbyupsphzjsq4.streamlit.app/)

---

# 🔄 System Workflow

```text
Side-Scan Sonar Image
        ↓
Image Input
        ↓
YOLO11n-Seg
        ↓
Detection + Segmentation
        ↓
Confidence Filtering
        ↓
GPS / Location Information
        ↓
Annotated Image
        ↓
JSON + CSV Reports
        ↓
Streamlit Dashboard


