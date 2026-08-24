import streamlit as st
from ultralytics import YOLO
from PIL import Image
import pandas as pd
import numpy as np
import json
import os
import tempfile
import io
import base64


# ============================================================
# AQUADEX AI
# AI-POWERED UNDERWATER MARINE ANOMALY DETECTION
# ============================================================

st.set_page_config(
    page_title="Aquadex AI",
    page_icon="🌊",
    layout="wide"
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_VERSION = "sss_shipwreck_v2_1024"
IMAGE_SIZE = 1024
DEFAULT_CONFIDENCE = 0.50


# ============================================================
# PAGE STYLE
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 0px;
    }

    .subtitle {
        font-size: 18px;
        margin-bottom: 25px;
    }

    .result-box {
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #cccccc;
        margin-top: 15px;
    }

    .warning-box {
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #e0a800;
        margin-top: 15px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="main-title">🌊 Aquadex AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
    AI-Powered Automated Underwater Marine Debris and
    Anomaly Detection using Side-Scan Sonar Imagery
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_model():

    possible_paths = [

        # Google Colab / Google Drive
        "/content/drive/MyDrive/Marine_Debris_Project/training/sss_shipwreck_v2_1024/weights/best.pt",

        "/content/drive/MyDrive/Marine_Debris_Project/training/sss_shipwreck_v1/weights/best.pt",

        # Local GitHub / deployment paths
        "best.pt",
        "model/best.pt",
        "./best.pt",
        "./model/best.pt"
    ]

    for path in possible_paths:

        if os.path.exists(path):

            try:
                return YOLO(path), path
            except Exception:
                continue

    return None, None


model, model_path = load_model()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("⚙️ Detection Settings")

confidence_threshold = st.sidebar.slider(
    "Confidence threshold",
    min_value=0.05,
    max_value=0.95,
    value=DEFAULT_CONFIDENCE,
    step=0.05
)

image_size = st.sidebar.selectbox(
    "Image size",
    [640, 768, 1024],
    index=2
)

st.sidebar.markdown("---")

st.sidebar.write(
    f"**Model:** YOLO11n-Seg"
)

st.sidebar.write(
    f"**Version:** {MODEL_VERSION}"
)

st.sidebar.write(
    f"**Image size:** {image_size}"
)


# ============================================================
# MODEL STATUS
# ============================================================

if model is None:

    st.error(
        """
        ⚠️ YOLO model weights were not found.

        Place `best.pt` in the application directory or
        configure the correct model path.
        """
    )

else:

    st.success(
        "✓ YOLO11n-Seg model loaded successfully"
    )


# ============================================================
# IMAGE UPLOAD
# ============================================================

st.header("📷 Sonar Image")

uploaded_file = st.file_uploader(
    "Upload a side-scan sonar image",
    type=["png", "jpg", "jpeg", "tif", "tiff"]
)


# ============================================================
# GPS INPUT
# ============================================================

st.header("📍 GPS / Location Information")

gps_col1, gps_col2 = st.columns(2)

with gps_col1:

    latitude = st.number_input(
        "Latitude",
        min_value=-90.0,
        max_value=90.0,
        value=0.0,
        format="%.8f"
    )

with gps_col2:

    longitude = st.number_input(
        "Longitude",
        min_value=-180.0,
        max_value=180.0,
        value=0.0,
        format="%.8f"
    )


gps_available = not (
    latitude == 0.0 and
    longitude == 0.0
)


# ============================================================
# MAP LINK
# ============================================================

def create_google_maps_url(lat, lon):

    return (
        f"https://www.google.com/maps"
        f"?q={lat},{lon}"
    )


# ============================================================
# REPORT CREATION
# ============================================================

def create_json_report(
    image_name,
    detections,
    latitude,
    longitude,
    confidence_threshold
):

    report = {

        "application": "Aquadex AI",

        "model": "YOLO11n-Seg",

        "model_version": MODEL_VERSION,

        "input_image": image_name,

        "confidence_threshold":
            confidence_threshold,

        "gps": {

            "latitude": latitude,

            "longitude": longitude,

            "source": (
                "User-provided GPS information"
                if gps_available
                else "Not provided"
            )
        },

        "summary": {

            "detections": len(detections),

            "candidate_detected":
                len(detections) > 0,

            "highest_confidence":
                max(
                    [d["confidence"] for d in detections],
                    default=0
                ),

            "average_confidence":
                (
                    sum(
                        d["confidence"]
                        for d in detections
                    ) / len(detections)
                    if detections
                    else 0
                )
        },

        "detections": detections
    }

    return report


# ============================================================
# RUN DETECTION
# ============================================================

if uploaded_file is not None:

    st.header("🔍 Analysis")

    image = Image.open(uploaded_file)

    st.subheader("Input Image")

    st.image(
        image,
        caption=uploaded_file.name,
        use_container_width=True
    )

    st.markdown("---")

    analyze_button = st.button(
        "🌊 Analyze with Aquadex AI",
        type="primary",
        use_container_width=True
    )

    if analyze_button:

        if model is None:

            st.error(
                "Model is unavailable. Cannot run inference."
            )

            st.stop()

        # ----------------------------------------------------
        # SAVE TEMPORARY IMAGE
        # ----------------------------------------------------

        suffix = os.path.splitext(
            uploaded_file.name
        )[1]

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        ) as temp_file:

            temp_file.write(
                uploaded_file.getbuffer()
            )

            temp_image_path = temp_file.name


        # ----------------------------------------------------
        # RUN YOLO
        # ----------------------------------------------------

        with st.spinner(
            "Running Aquadex AI detection..."
        ):

            try:

                results = model.predict(
                    source=temp_image_path,
                    conf=confidence_threshold,
                    imgsz=image_size,
                    verbose=False
                )

            except Exception as e:

                st.error(
                    f"Inference failed: {e}"
                )

                st.stop()


        result = results[0]


        # ----------------------------------------------------
        # EXTRACT DETECTIONS
        # ----------------------------------------------------

        detections = []

        boxes = result.boxes

        if boxes is not None:

            for i in range(len(boxes)):

                confidence = float(
                    boxes.conf[i].item()
                )

                class_id = int(
                    boxes.cls[i].item()
                )

                if result.names:

                    class_name = result.names.get(
                        class_id,
                        str(class_id)
                    )

                else:

                    class_name = str(class_id)


                xyxy = boxes.xyxy[i].cpu().numpy()

                x1 = float(xyxy[0])
                y1 = float(xyxy[1])
                x2 = float(xyxy[2])
                y2 = float(xyxy[3])

                width = x2 - x1
                height = y2 - y1

                detection = {

                    "detection_id":
                        i + 1,

                    "class_name":
                        class_name,

                    "confidence":
                        round(confidence, 4),

                    "latitude":
                        latitude
                        if gps_available
                        else None,

                    "longitude":
                        longitude
                        if gps_available
                        else None,

                    "x1":
                        round(x1, 2),

                    "y1":
                        round(y1, 2),

                    "x2":
                        round(x2, 2),

                    "y2":
                        round(y2, 2),

                    "width":
                        round(width, 2),

                    "height":
                        round(height, 2)
                }

                detections.append(
                    detection
                )


        # ----------------------------------------------------
        # SUMMARY VALUES
        # ----------------------------------------------------

        number_detections = len(
            detections
        )

        confidence_values = [
            d["confidence"]
            for d in detections
        ]

        highest_confidence = (
            max(confidence_values)
            if confidence_values
            else 0
        )

        average_confidence = (
            sum(confidence_values)
            / len(confidence_values)
            if confidence_values
            else 0
        )


        # ----------------------------------------------------
        # RESULT HEADER
        # ----------------------------------------------------

        st.markdown("---")

        st.header(
            "🌊 Aquadex AI Result"
        )


        # ----------------------------------------------------
        # METRIC CARDS
        # ----------------------------------------------------

        c1, c2, c3 = st.columns(3)

        with c1:

            st.metric(
                "Detections",
                number_detections
            )

        with c2:

            st.metric(
                "Highest Confidence",
                f"{highest_confidence * 100:.1f}%"
            )

        with c3:

            st.metric(
                "Average Confidence",
                f"{average_confidence * 100:.1f}%"
            )


        # ----------------------------------------------------
        # DETECTION STATUS
        # ----------------------------------------------------

        if number_detections > 0:

            st.success(
                "✓ SHIPWRECK / MAN-MADE ANOMALY "
                "CANDIDATE DETECTED"
            )

        else:

            st.info(
                "No candidate anomaly detected "
                "above the selected confidence threshold."
            )


        # ----------------------------------------------------
        # ANNOTATED IMAGE
        # ----------------------------------------------------

        st.subheader(
            "🎯 Detection / Segmentation Result"
        )

        try:

            annotated_array = result.plot(
                labels=True,
                boxes=True,
                masks=True
            )

            annotated_image = Image.fromarray(
                annotated_array[..., ::-1]
            )

            st.image(
                annotated_image,
                caption="Aquadex AI Annotated Result",
                use_container_width=True
            )

        except Exception:

            st.warning(
                "Annotated image could not be generated."
            )

            annotated_image = image


        # ----------------------------------------------------
        # GPS RESULT
        # ----------------------------------------------------

        st.subheader(
            "📍 Detection Location"
        )

        if gps_available:

            gps1, gps2 = st.columns(2)

            with gps1:

                st.metric(
                    "Latitude",
                    f"{latitude:.8f}"
                )

            with gps2:

                st.metric(
                    "Longitude",
                    f"{longitude:.8f}"
                )


            maps_url = create_google_maps_url(
                latitude,
                longitude
            )

            st.link_button(
                "🗺️ Open Location in Google Maps",
                maps_url,
                use_container_width=True
            )

        else:

            st.warning(
                """
                GPS coordinates were not provided.
                Enter latitude and longitude above to
                generate a geotagged result.
                """
            )


        # ----------------------------------------------------
        # INDIVIDUAL DETECTIONS
        # ----------------------------------------------------

        st.subheader(
            "📊 Individual Detections"
        )

        if detections:

            detection_table = pd.DataFrame(
                detections
            )

            display_columns = [

                "detection_id",

                "class_name",

                "confidence",

                "latitude",

                "longitude",

                "width",

                "height"
            ]

            st.dataframe(
                detection_table[
                    display_columns
                ],
                use_container_width=True
            )

        else:

            st.write(
                "No detections available."
            )


        # ----------------------------------------------------
        # JSON REPORT
        # ----------------------------------------------------

        report = create_json_report(

            image_name=uploaded_file.name,

            detections=detections,

            latitude=latitude,

            longitude=longitude,

            confidence_threshold=
                confidence_threshold
        )


        json_data = json.dumps(
            report,
            indent=4
        )


        # ----------------------------------------------------
        # CSV REPORT
        # ----------------------------------------------------

        if detections:

            csv_dataframe = pd.DataFrame(
                detections
            )

        else:

            csv_dataframe = pd.DataFrame(
                columns=[
                    "detection_id",
                    "class_name",
                    "confidence",
                    "latitude",
                    "longitude",
                    "x1",
                    "y1",
                    "x2",
                    "y2",
                    "width",
                    "height"
                ]
            )


        csv_data = csv_dataframe.to_csv(
            index=False
        )


        # ----------------------------------------------------
        # DOWNLOADS
        # ----------------------------------------------------

        st.subheader(
            "📥 Export Reports"
        )

        download1, download2, download3 = st.columns(3)


        with download1:

            st.download_button(
                label="📄 Download JSON",
                data=json_data,
                file_name=(
                    "aquadex_report.json"
                ),
                mime="application/json",
                use_container_width=True
            )


        with download2:

            st.download_button(
                label="📊 Download CSV",
                data=csv_data,
                file_name=(
                    "aquadex_report.csv"
                ),
                mime="text/csv",
                use_container_width=True
            )


        # ----------------------------------------------------
        # ANNOTATED IMAGE DOWNLOAD
        # ----------------------------------------------------

        with download3:

            image_buffer = io.BytesIO()

            annotated_image.save(
                image_buffer,
                format="PNG"
            )

            image_bytes = (
                image_buffer.getvalue()
            )

            st.download_button(
                label="🖼️ Download Image",
                data=image_bytes,
                file_name=(
                    "aquadex_annotated.png"
                ),
                mime="image/png",
                use_container_width=True
            )


        # ----------------------------------------------------
        # JSON PREVIEW
        # ----------------------------------------------------

        with st.expander(
            "View JSON Report"
        ):

            st.json(report)


        # ----------------------------------------------------
        # PROTOTYPE DISCLAIMER
        # ----------------------------------------------------

        st.markdown("---")

        st.warning(
            """
            ⚠️ This is an AI-assisted research prototype.
            A candidate detection is not confirmation of a
            shipwreck or other underwater object and requires
            human verification.
            """
        )


        # ----------------------------------------------------
        # CLEAN TEMP FILE
        # ----------------------------------------------------

        try:

            os.remove(
                temp_image_path
            )

        except Exception:

            pass


# ============================================================
# NO IMAGE STATE
# ============================================================

else:

    st.info(
        """
        👆 Upload a side-scan sonar image above to begin
        Aquadex AI analysis.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "🌊 Aquadex AI — Side-Scan Sonar Anomaly Detection "
    "Research Prototype"
)
