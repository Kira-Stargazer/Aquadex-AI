import os
import io
import json
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from ultralytics import YOLO


# ============================================================
# CONFIGURATION
# ============================================================

import os

APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(APP_DIR)

MODEL_PATH = os.path.join(
    PROJECT_DIR,
    "model",
    "best.pt"
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Aquadex AI",
    page_icon="🌊",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        opacity: 0.75;
        margin-bottom: 25px;
    }

    .status-box {
        padding: 18px;
        border-radius: 12px;
        text-align: center;
        font-size: 24px;
        font-weight: 800;
        margin-top: 20px;
        margin-bottom: 25px;
    }

    .detected {
        background-color: #0b4d27;
        color: white;
    }

    .not-detected {
        background-color: #5a1c1c;
        color: white;
    }

    .gps-box {
        padding: 18px;
        border-radius: 12px;
        border: 1px solid rgba(128,128,128,0.3);
        margin-top: 10px;
        margin-bottom: 20px;
    }

    .gps-value {
        font-size: 22px;
        font-weight: 700;
    }

    .warning-box {
        padding: 18px;
        border-radius: 10px;
        background-color: #4b4a12;
        color: white;
        margin-top: 20px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_PATH):
        st.error(
            "YOLO model not found.\n\n"
            f"Expected location:\n{MODEL_PATH}\n\n"
            "Please upload best.pt into the model/ folder."
        )
        st.stop()

    return YOLO(MODEL_PATH)


model = load_model()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("⚙️ Analysis Settings")

confidence_threshold = st.sidebar.slider(
    "Confidence threshold",
    min_value=0.10,
    max_value=0.95,
    value=0.70,
    step=0.05
)

image_size = st.sidebar.selectbox(
    "Image size",
    [640, 768, 1024],
    index=2
)

st.sidebar.divider()

st.sidebar.title("🧠 AI Model")

st.sidebar.write("YOLO11n-Seg")
st.sidebar.write("Shipwreck / anomaly segmentation")
st.sidebar.write("Model: sss_shipwreck_v2_1024")

st.sidebar.divider()

st.sidebar.title("📍 GPS")

gps_latitude = st.sidebar.number_input(
    "Latitude",
    min_value=-90.0,
    max_value=90.0,
    value=0.0,
    step=0.000001,
    format="%.6f"
)

gps_longitude = st.sidebar.number_input(
    "Longitude",
    min_value=-180.0,
    max_value=180.0,
    value=0.0,
    step=0.000001,
    format="%.6f"
)

gps_source = st.sidebar.selectbox(
    "GPS source",
    [
        "Manual GPS input",
        "Sonar metadata",
        "External navigation system"
    ]
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🌊 Aquadex AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-Powered Automated Underwater Marine Debris and '
    'Anomaly Detection using Side-Scan Sonar Imagery'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# SYSTEM STATUS
# ============================================================

with st.sidebar:

    st.success("AI model loaded")

    st.write("Model:")
    st.code("YOLO11n-Seg")

    st.write("Task:")
    st.code("Detection + Segmentation")


# ============================================================
# GPS DISPLAY
# ============================================================

st.subheader("📍 GPS Location")

gps_col1, gps_col2, gps_col3 = st.columns(3)

with gps_col1:

    st.metric(
        "Latitude",
        f"{gps_latitude:.6f}"
    )

with gps_col2:

    st.metric(
        "Longitude",
        f"{gps_longitude:.6f}"
    )

with gps_col3:

    st.metric(
        "GPS Source",
        gps_source
    )


if gps_latitude != 0.0 or gps_longitude != 0.0:

    google_maps_url = (
        "https://www.google.com/maps/search/?api=1"
        f"&query={gps_latitude},{gps_longitude}"
    )

    st.markdown(
        f"""
        <div class="gps-box">

        <div class="gps-value">
        📍 {gps_latitude:.6f}, {gps_longitude:.6f}
        </div>

        <br>

        <a href="{google_maps_url}" target="_blank">
        🌍 Open GPS Location in Google Maps
        </a>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# IMAGE UPLOAD
# ============================================================

st.subheader("📷 Input Sonar Image")

uploaded_file = st.file_uploader(
    "Upload a side-scan sonar image",
    type=[
        "png",
        "jpg",
        "jpeg",
        "tif",
        "tiff"
    ]
)


# ============================================================
# ANALYSIS
# ============================================================

if uploaded_file is not None:

    image_bytes = uploaded_file.read()

    input_image = Image.open(
        io.BytesIO(image_bytes)
    ).convert("RGB")

    st.subheader("🔍 Sonar Image")

    st.image(
        input_image,
        caption="Input Side-Scan Sonar Image",
        use_container_width=True
    )

    analyze = st.button(
        "🌊 ANALYZE WITH AQUADEX AI",
        type="primary",
        use_container_width=True
    )

    if analyze:

        with st.spinner(
            "Running YOLO sonar analysis..."
        ):

            image_array = np.array(input_image)

            results = model.predict(
                source=image_array,
                conf=confidence_threshold,
                imgsz=image_size,
                verbose=False
            )

            result = results[0]

            # =================================================
            # DETECTIONS
            # =================================================

            detections = []

            names = result.names

            if result.boxes is not None:

                boxes = result.boxes

                for i in range(len(boxes)):

                    confidence = float(
                        boxes.conf[i].item()
                    )

                    class_id = int(
                        boxes.cls[i].item()
                    )

                    class_name = names.get(
                        class_id,
                        str(class_id)
                    )

                    xyxy = (
                        boxes.xyxy[i]
                        .cpu()
                        .numpy()
                    )

                    x1, y1, x2, y2 = [
                        float(v)
                        for v in xyxy
                    ]

                    width = x2 - x1
                    height = y2 - y1

                    detection = {

                        "detection_id":
                            i + 1,

                        "class_id":
                            class_id,

                        "class_name":
                            class_name,

                        "confidence":
                            round(
                                confidence,
                                4
                            ),

                        "confidence_percent":
                            round(
                                confidence * 100,
                                2
                            ),

                        "x1":
                            round(x1, 2),

                        "y1":
                            round(y1, 2),

                        "x2":
                            round(x2, 2),

                        "y2":
                            round(y2, 2),

                        "width_pixels":
                            round(width, 2),

                        "height_pixels":
                            round(height, 2),

                        "latitude":
                            gps_latitude,

                        "longitude":
                            gps_longitude
                    }

                    detections.append(
                        detection
                    )


            # =================================================
            # ANNOTATED IMAGE
            # =================================================

            annotated_array = result.plot()

            annotated_image = Image.fromarray(
                annotated_array[..., ::-1]
            )


            # =================================================
            # SUMMARY
            # =================================================

            highest_confidence = 0.0

            if detections:

                highest_confidence = max(
                    d["confidence"]
                    for d in detections
                )


            # =================================================
            # REPORT
            # =================================================

            timestamp = datetime.now().strftime(
                "%Y%m%d_%H%M%S"
            )

            report = {

                "application":
                    "Aquadex AI",

                "project":
                    "AI-Powered Automated Underwater "
                    "Marine Debris and Anomaly Detection "
                    "System using Side-Scan Sonar Imagery",

                "model":
                    "YOLO11n-Seg",

                "model_version":
                    "sss_shipwreck_v2_1024",

                "task":
                    "Detection + Segmentation",

                "image":
                    uploaded_file.name,

                "image_width":
                    input_image.width,

                "image_height":
                    input_image.height,

                "image_size":
                    image_size,

                "confidence_threshold":
                    confidence_threshold,

                "timestamp":
                    timestamp,

                "gps": {

                    "latitude":
                        gps_latitude,

                    "longitude":
                        gps_longitude,

                    "source":
                        gps_source
                },

                "summary": {

                    "detections":
                        len(detections),

                    "highest_confidence":
                        highest_confidence,

                    "highest_confidence_percent":
                        round(
                            highest_confidence * 100,
                            2
                        ),

                    "candidate_detected":
                        len(detections) > 0
                },

                "detections":
                    detections
            }


            # =================================================
            # SESSION STATE
            # =================================================

            st.session_state["report"] = report

            st.session_state[
                "annotated_image"
            ] = annotated_image

            st.session_state[
                "detections"
            ] = detections

            st.success(
                "Analysis completed successfully."
            )


# ============================================================
# DISPLAY RESULTS
# ============================================================

if "report" in st.session_state:

    report = st.session_state["report"]

    detections = st.session_state["detections"]

    annotated_image = (
        st.session_state["annotated_image"]
    )

    st.divider()


    # ========================================================
    # STATUS
    # ========================================================

    if len(detections) > 0:

        st.markdown(
            '<div class="status-box detected">'
            '🟢 SHIPWRECK / ANOMALY CANDIDATE DETECTED'
            '</div>',
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            '<div class="status-box not-detected">'
            '🔴 NO CANDIDATE DETECTED'
            '</div>',
            unsafe_allow_html=True
        )


    # ========================================================
    # DETECTION IMAGE
    # ========================================================

    st.subheader("🧠 Aquadex AI Detection")

    st.image(
        annotated_image,
        caption="YOLO11n-Seg Annotated Detection",
        use_container_width=True
    )


    # ========================================================
    # SUMMARY
    # ========================================================

    st.subheader("📊 Analysis Summary")

    metric1, metric2, metric3 = st.columns(3)

    with metric1:

        st.metric(
            "Candidates",
            len(detections)
        )

    with metric2:

        st.metric(
            "Highest Confidence",
            f"{report['summary']['highest_confidence'] * 100:.1f}%"
        )

    with metric3:

        st.metric(
            "Threshold",
            f"{report['confidence_threshold'] * 100:.0f}%"
        )


    # ========================================================
    # GPS RESULT
    # ========================================================

    st.subheader("📍 GPS Output")

    gps1, gps2, gps3 = st.columns(3)

    with gps1:

        st.metric(
            "Latitude",
            f"{report['gps']['latitude']:.6f}"
        )

    with gps2:

        st.metric(
            "Longitude",
            f"{report['gps']['longitude']:.6f}"
        )

    with gps3:

        st.metric(
            "GPS Source",
            report["gps"]["source"]
        )


    if (
        report["gps"]["latitude"] != 0.0
        or
        report["gps"]["longitude"] != 0.0
    ):

        maps_url = (
            "https://www.google.com/maps/search/?api=1"
            f"&query="
            f"{report['gps']['latitude']},"
            f"{report['gps']['longitude']}"
        )

        st.markdown(
            f"""
            ### 🌍 Detected Location

            **GPS Coordinates**

            `{report['gps']['latitude']:.6f},`
            `{report['gps']['longitude']:.6f}`

            **Source**

            `{report['gps']['source']}`

            [📍 Open Location in Google Maps]({maps_url})
            """
        )


    # ========================================================
    # INDIVIDUAL DETECTIONS
    # ========================================================

    st.subheader("🔎 Individual Detections")

    if detections:

        detection_df = pd.DataFrame(
            detections
        )

        st.dataframe(
            detection_df,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "No detections above the selected "
            "confidence threshold."
        )


    # ========================================================
    # EXPORT
    # ========================================================

    st.subheader("📥 Export Reports")

    json_bytes = json.dumps(
        report,
        indent=4
    ).encode("utf-8")


    if detections:

        csv_df = pd.DataFrame(
            detections
        )

    else:

        csv_df = pd.DataFrame(
            [{
                "latitude":
                    report["gps"]["latitude"],

                "longitude":
                    report["gps"]["longitude"],

                "result":
                    "No detection"
            }]
        )


    csv_bytes = (
        csv_df
        .to_csv(index=False)
        .encode("utf-8")
    )


    image_buffer = io.BytesIO()

    annotated_image.save(
        image_buffer,
        format="PNG"
    )

    image_bytes = (
        image_buffer.getvalue()
    )


    download1, download2, download3 = (
        st.columns(3)
    )


    with download1:

        st.download_button(
            "📄 Download JSON",
            data=json_bytes,
            file_name="aquadex_report.json",
            mime="application/json",
            use_container_width=True
        )


    with download2:

        st.download_button(
            "📊 Download CSV",
            data=csv_bytes,
            file_name="aquadex_report.csv",
            mime="text/csv",
            use_container_width=True
        )


    with download3:

        st.download_button(
            "🖼️ Download Image",
            data=image_bytes,
            file_name="aquadex_annotated.png",
            mime="image/png",
            use_container_width=True
        )


    # ========================================================
    # AI INTERPRETATION
    # ========================================================

    st.subheader("🧠 AI Interpretation")

    if detections:

        st.write(
            f"Aquadex AI identified "
            f"**{len(detections)} potential "
            f"shipwreck/anomaly region(s)** "
            f"above the selected confidence threshold."
        )

        st.write(
            f"The highest-confidence candidate "
            f"has a confidence of "
            f"**{report['summary']['highest_confidence'] * 100:.1f}%**."
        )

    else:

        st.write(
            "Aquadex AI did not identify a candidate "
            "above the selected confidence threshold."
        )


    st.markdown(
        """
        <div class="warning-box">

        ⚠️ <b>Research prototype:</b>

        An AI detection represents a candidate region
        and is not confirmation of a shipwreck,
        archaeological site, or marine object.

        Human verification is required.

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🌊 Aquadex AI • AI-assisted sonar analysis "
    "for underwater marine exploration • Research Prototype"
)
