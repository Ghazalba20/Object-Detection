import os
import tempfile
from collections import Counter

import cv2
import streamlit as st
from PIL import Image
from ultralytics import YOLO


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="VisionAI | Object Detection",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* =========================
       GENERAL
       ========================= */

    .stApp {
        background: #0b1020;
        color: #f8fafc;
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    .block-container {
        max-width: 1400px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }


    /* =========================
       SIDEBAR
       ========================= */

    section[data-testid="stSidebar"] {
        background: #111827;
        border-right: 1px solid #263244;
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #ffffff;
    }


    /* =========================
       HERO
       ========================= */

    .hero {
        background: linear-gradient(
            135deg,
            #111827,
            #172554
        );

        border: 1px solid #263b65;
        border-radius: 24px;
        padding: 32px;
        margin-bottom: 28px;

        box-shadow:
            0 15px 40px rgba(0, 0, 0, 0.25);
    }

    .hero-title {
        font-size: 38px;
        font-weight: 800;
        margin-bottom: 8px;
        color: #ffffff;
    }

    .hero-subtitle {
        font-size: 16px;
        color: #aab7cf;
        line-height: 1.7;
    }

    .badge {
        display: inline-block;
        background: #1e3a8a;
        color: #bfdbfe;
        padding: 6px 14px;
        border-radius: 999px;
        font-size: 13px;
        font-weight: 600;
        margin-bottom: 15px;
    }


    /* =========================
       DETECTION CARDS
       ========================= */

    .detection-card {
        background: #111827;
        border: 1px solid #263244;
        border-radius: 18px;
        padding: 20px;
        margin-bottom: 12px;

        transition:
            transform 0.2s ease,
            border-color 0.2s ease;
    }

    .detection-card:hover {
        transform: translateY(-3px);
        border-color: #3b82f6;
    }

    .detection-icon {
        font-size: 30px;
        margin-bottom: 8px;
    }

    .detection-name {
        font-size: 15px;
        color: #94a3b8;
    }

    .detection-count {
        font-size: 28px;
        font-weight: 800;
        color: #ffffff;
    }


    /* =========================
       SECTION TITLES
       ========================= */

    .section-title {
        font-size: 24px;
        font-weight: 750;
        color: #ffffff;
        margin-top: 25px;
        margin-bottom: 15px;
    }


    /* =========================
       INFO CARD
       ========================= */

    .info-card {
        background: #0f172a;
        border: 1px solid #263244;
        border-radius: 18px;
        padding: 20px;
        color: #cbd5e1;
        line-height: 1.8;
    }


    /* =========================
       UPLOAD BOX
       ========================= */

    [data-testid="stFileUploader"] {
        background: #111827;
        border-radius: 18px;
        padding: 10px;
    }


    /* =========================
       BUTTONS
       ========================= */

    .stButton > button {
        width: 100%;
        border-radius: 12px;
        border: none;
        padding: 12px 20px;
        font-weight: 700;

        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
    }


    /* =========================
       RADIO
       ========================= */

    div[role="radiogroup"] {
        gap: 10px;
    }


    /* =========================
       METRIC
       ========================= */

    [data-testid="stMetric"] {
        background: #111827;
        border: 1px solid #263244;
        border-radius: 16px;
        padding: 15px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")


model = load_model()


# ============================================================
# CLASS ICONS
# ============================================================

CLASS_ICONS = {
    "person": "👤",

    "car": "🚗",
    "bus": "🚌",
    "truck": "🚚",
    "motorcycle": "🏍️",
    "bicycle": "🚲",
    "train": "🚆",
    "airplane": "✈️",
    "boat": "🚤",

    "dog": "🐕",
    "cat": "🐈",
    "bird": "🐦",
    "horse": "🐎",
    "sheep": "🐑",
    "cow": "🐄",
    "elephant": "🐘",
    "bear": "🐻",
    "zebra": "🦓",
    "giraffe": "🦒",

    "backpack": "🎒",
    "umbrella": "☂️",
    "handbag": "👜",
    "tie": "👔",
    "suitcase": "🧳",

    "bottle": "🍼",
    "wine glass": "🍷",
    "cup": "☕",
    "fork": "🍴",
    "knife": "🔪",
    "spoon": "🥄",
    "bowl": "🥣",

    "banana": "🍌",
    "apple": "🍎",
    "sandwich": "🥪",
    "orange": "🍊",
    "broccoli": "🥦",
    "carrot": "🥕",
    "pizza": "🍕",
    "donut": "🍩",
    "cake": "🍰",

    "chair": "🪑",
    "couch": "🛋️",
    "bed": "🛏️",
    "dining table": "🍽️",
    "toilet": "🚽",

    "tv": "📺",
    "laptop": "💻",
    "mouse": "🖱️",
    "remote": "🎮",
    "keyboard": "⌨️",
    "cell phone": "📱",
    "microwave": "♨️",
    "oven": "🔥",
    "refrigerator": "🧊",

    "book": "📚",
    "clock": "🕐",
    "vase": "🏺",
    "scissors": "✂️",
    "teddy bear": "🧸",
    "hair drier": "💨",
    "toothbrush": "🪥",
}


def get_icon(class_name):
    return CLASS_ICONS.get(
        class_name.lower(),
        "🔹",
    )


# ============================================================
# GET CLASS NAME
# ============================================================

def get_class_name(class_id):

    names = model.names

    if isinstance(names, dict):
        return names.get(
            class_id,
            str(class_id),
        )

    return names[class_id]


# ============================================================
# IMAGE DETECTION
# ============================================================

def detect_image(image, confidence):

    results = model.predict(
        source=image,
        conf=confidence,
        verbose=False,
    )

    result = results[0]

    annotated = result.plot()

    counter = Counter()

    if result.boxes is not None:

        for class_id in result.boxes.cls.tolist():

            class_id = int(class_id)

            class_name = get_class_name(
                class_id,
            )

            counter[class_name] += 1

    return annotated, counter


# ============================================================
# VIDEO PROCESSING
# ============================================================

def process_video(
    video_path,
    confidence,
    output_path,
    progress_bar,
):

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(
            "ویدیو قابل باز شدن نیست."
        )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    if fps <= 0:
        fps = 25

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (width, height),
    )

    if not writer.isOpened():

        cap.release()

        raise RuntimeError(
            "امکان ساخت فایل خروجی ویدیو وجود ندارد."
        )

    total_counter = Counter()

    processed_frames = 0

    try:

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            results = model.predict(
                source=frame,
                conf=confidence,
                verbose=False,
            )

            result = results[0]

            annotated_frame = result.plot()

            if result.boxes is not None:

                for class_id in result.boxes.cls.tolist():

                    class_id = int(class_id)

                    class_name = get_class_name(
                        class_id
                    )

                    total_counter[class_name] += 1

            writer.write(
                annotated_frame
            )

            processed_frames += 1

            if total_frames > 0:

                progress = min(
                    processed_frames / total_frames,
                    1.0,
                )

                progress_bar.progress(
                    progress
                )

    finally:

        cap.release()
        writer.release()

    return output_path, total_counter


# ============================================================
# SHOW DETECTION RESULTS
# ============================================================

def show_detection_results(counter):

    st.markdown(
        '<div class="section-title">'
        '📊 Detection Results'
        '</div>',
        unsafe_allow_html=True,
    )

    if not counter:

        st.warning(
            "هیچ آبجکتی پیدا نشد."
        )

        return

    sorted_items = sorted(
        counter.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    # --------------------------------------------------------
    # TOTAL OBJECTS
    # --------------------------------------------------------

    total_objects = sum(
        counter.values()
    )

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "🔍 Total Objects",
            total_objects,
        )

    with col2:

        st.metric(
            "🏷️ Object Classes",
            len(counter),
        )

    st.markdown(
        "<br>",
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # OBJECT CARDS
    # --------------------------------------------------------

    columns = st.columns(
        min(len(sorted_items), 4)
    )

    for index, (name, count) in enumerate(
        sorted_items
    ):

        with columns[
            index % len(columns)
        ]:

            icon = get_icon(name)

            st.markdown(
                f"""
                <div class="detection-card">

                    <div class="detection-icon">
                        {icon}
                    </div>

                    <div class="detection-name">
                        {name}
                    </div>

                    <div class="detection-count">
                        {count}
                    </div>

                    <div style="
                        color:#64748b;
                        font-size:12px;
                    ">
                        detected
                    </div>

                </div>
                """,
                unsafe_allow_html=True,
            )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">'
        '📝 Summary'
        '</div>',
        unsafe_allow_html=True,
    )

    summary_text = "، ".join(
        [
            f"{get_icon(name)} {count} تا {name}"
            for name, count in sorted_items
        ]
    )

    st.markdown(
        f"""
        <div class="info-card">

            در این ورودی مجموعاً
            <b>{total_objects}</b>
            آبجکت در
            <b>{len(counter)}</b>
            کلاس مختلف شناسایی شد.

            <br><br>

            {summary_text}

        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# HERO HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">

        <div class="badge">
            ✨ AI COMPUTER VISION
        </div>

        <div class="hero-title">
            👁️ VisionAI
        </div>

        <div class="hero-subtitle">

            سیستم تشخیص اشیاء با استفاده از
            <b>YOLOv8</b>

            <br>

            تشخیص سریع و هوشمند اشیاء در عکس،
            ویدیو و دوربین

        </div>

    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        # ⚙️ Settings

        تنظیمات تشخیص را از این بخش کنترل کن.
        """
    )

    st.divider()

    confidence = st.slider(
        "🎯 Confidence",
        min_value=0.05,
        max_value=0.95,
        value=0.35,
        step=0.05,
    )

    st.caption(
        f"Confidence فعلی: {confidence:.2f}"
    )

    st.info(
        "Confidence بالاتر باعث می‌شود "
        "مدل فقط تشخیص‌های مطمئن‌تر را نمایش دهد."
    )

    st.divider()

    st.markdown(
        """
        ### 🤖 Model

        **YOLOv8 Nano**

        مناسب برای اجرای سریع و سبک.
        """
    )

    st.divider()

    st.caption(
        "VisionAI • YOLO Object Detection"
    )


# ============================================================
# INPUT TYPE
# ============================================================

st.markdown(
    '<div class="section-title">'
    '📥 Choose Input'
    '</div>',
    unsafe_allow_html=True,
)

input_type = st.radio(
    "نوع ورودی:",
    [
        "🖼️ عکس",
        "🎥 ویدیو",
        "📹 وب‌کم",
    ],
    horizontal=True,
    label_visibility="collapsed",
)


# ============================================================
# IMAGE
# ============================================================

if input_type == "🖼️ عکس":

    uploaded_file = st.file_uploader(
        "📤 تصویر خود را انتخاب کنید",
        type=[
            "jpg",
            "jpeg",
            "png",
            "webp",
            "bmp",
        ],
    )

    if uploaded_file is not None:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

        st.markdown(
            '<div class="section-title">'
            '🖼️ Image Preview'
            '</div>',
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns(
            2,
            gap="large",
        )

        with col1:

            st.image(
                image,
                caption="Original Image",
                use_container_width=True,
            )

        with col2:

            if st.button(
                "🔍 Start Detection",
                type="primary",
            ):

                with st.spinner(
                    "در حال تحلیل تصویر..."
                ):

                    annotated, counter = detect_image(
                        image,
                        confidence,
                    )

                st.image(
                    annotated,
                    caption="Detection Result",
                    channels="BGR",
                    use_container_width=True,
                )

                show_detection_results(
                    counter
                )


# ============================================================
# VIDEO
# ============================================================

elif input_type == "🎥 ویدیو":

    uploaded_video = st.file_uploader(
        "📤 ویدیوی خود را انتخاب کنید",
        type=[
            "mp4",
            "avi",
            "mov",
            "mkv",
        ],
    )

    if uploaded_video is not None:

        input_temp = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=os.path.splitext(
                uploaded_video.name
            )[1],
        )

        input_temp.write(
            uploaded_video.getbuffer()
        )

        input_temp.close()

        output_temp = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4",
        )

        output_temp.close()

        st.markdown(
            '<div class="section-title">'
            '🎥 Original Video'
            '</div>',
            unsafe_allow_html=True,
        )

        st.video(
            uploaded_video
        )

        if st.button(
            "🎯 Start Video Detection",
            type="primary",
        ):

            progress_bar = st.progress(0)

            try:

                with st.spinner(
                    "در حال پردازش ویدیو..."
                ):

                    output_path, counter = process_video(
                        input_temp.name,
                        confidence,
                        output_temp.name,
                        progress_bar,
                    )

                progress_bar.progress(1.0)

                st.success(
                    "ویدیو با موفقیت پردازش شد ✅"
                )

                st.markdown(
                    '<div class="section-title">'
                    '🎬 Processed Video'
                    '</div>',
                    unsafe_allow_html=True,
                )

                with open(
                    output_path,
                    "rb",
                ) as video_file:

                    video_bytes = video_file.read()

                st.video(
                    video_bytes
                )

                st.download_button(
                    label="⬇️ Download Result",
                    data=video_bytes,
                    file_name="detected_video.mp4",
                    mime="video/mp4",
                )

                show_detection_results(
                    counter
                )

            except Exception as e:

                st.error(
                    f"❌ خطا در پردازش ویدیو:\n\n{e}"
                )

            finally:

                if os.path.exists(
                    input_temp.name
                ):

                    os.remove(
                        input_temp.name
                    )

                if os.path.exists(
                    output_temp.name
                ):

                    os.remove(
                        output_temp.name
                    )


# ============================================================
# WEBCAM
# ============================================================

elif input_type == "📹 وب‌کم":

    st.markdown(
        '<div class="section-title">'
        '📷 Camera'
        '</div>',
        unsafe_allow_html=True,
    )

    st.info(
        "با دوربین یک عکس بگیر تا YOLO اشیاء موجود "
        "در تصویر را شناسایی کند."
    )

    camera_image = st.camera_input(
        "📸 Take a Picture"
    )

    if camera_image is not None:

        image = Image.open(
            camera_image
        ).convert("RGB")

        with st.spinner(
            "در حال تشخیص..."
        ):

            annotated, counter = detect_image(
                image,
                confidence,
            )

        col1, col2 = st.columns(
            2,
            gap="large",
        )

        with col1:

            st.image(
                image,
                caption="Camera Image",
                use_container_width=True,
            )

        with col2:

            st.image(
                annotated,
                caption="Detection Result",
                channels="BGR",
                use_container_width=True,
            )

        show_detection_results(
            counter
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <br><br>

    <div style="
        text-align:center;
        color:#64748b;
        padding:25px;
        border-top:1px solid #263244;
    ">

        👁️ <b>VisionAI</b>
        &nbsp;•&nbsp;
        Powered by YOLOv8
        &nbsp;•&nbsp;
        Computer Vision

    </div>
    """,
    unsafe_allow_html=True,
)