import cv2 as cv
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av

# Set up Streamlit page configuration
st.set_page_config(page_title="SNAPnFIX", page_icon="📄", layout="wide")

# App styling
st.markdown("""
    <style>
    .main-title {
        color: #ffffff;
        font-size: 36px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 20px;
    }
    .header {
        color: #ffffff;
        background-color: #4CAF50;
        padding: 10px;
        border-radius: 10px;
        font-size: 20px;
        text-align: center;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">📄 SNAPnFIX: Document Scanner and Fixer</div>', unsafe_allow_html=True)

# Parameters for document detection
contour_area_threshold = 0.3 * 640 * 480

# Preprocess the image
def pre_process_image(img):
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img_blur = cv.GaussianBlur(img_gray, (5, 5), 0)
    img_canny = cv.Canny(img_blur, 75, 150)
    kernel = np.ones((5, 5))
    img_dilated = cv.dilate(img_canny, kernel, iterations=2)
    img_eroded = cv.erode(img_dilated, kernel, iterations=1)
    return img_eroded

# Detect document contours
def find_document_contours(img):
    contours, _ = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv.contourArea(cnt)
        if area > contour_area_threshold:
            peri = cv.arcLength(cnt, True)
            approx = cv.approxPolyDP(cnt, 0.02 * peri, True)
            if len(approx) == 4:
                return approx
    return None

# Reorder points for perspective transformation
def reorder_points(points):
    points = points.reshape((4, 2))
    new_points = np.zeros((4, 1, 2), np.int32)
    add = points.sum(axis=1)
    new_points[0] = points[np.argmin(add)]  # Top-left
    new_points[3] = points[np.argmax(add)]  # Bottom-right
    diff = np.diff(points, axis=1)
    new_points[1] = points[np.argmin(diff)]  # Top-right
    new_points[2] = points[np.argmax(diff)]  # Bottom-left
    return new_points

# Apply perspective transform to warp the document
def warp_document(img, points):
    points = reorder_points(points)
    pts1 = np.float32(points)
    x, y, w, h = cv.boundingRect(points)
    pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    matrix = cv.getPerspectiveTransform(pts1, pts2)
    img_warped = cv.warpPerspective(img, matrix, (w, h))

    # Sharpen the image
    sharpening_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    img_sharpened = cv.filter2D(img_warped, -1, sharpening_kernel)
    return img_sharpened

# Video processing class
class DocumentScanner(VideoProcessorBase):
    def __init__(self):
        self.result_img = None

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        processed_img = pre_process_image(img)
        contours = find_document_contours(processed_img)

        if contours is not None:
            self.result_img = warp_document(img, contours)
            return av.VideoFrame.from_ndarray(cv.cvtColor(self.result_img, cv.COLOR_BGR2RGB), format="rgb24")
        else:
            return av.VideoFrame.from_ndarray(cv.cvtColor(img, cv.COLOR_BGR2RGB), format="rgb24")

# Main layout with columns for webcam and upload functionality
col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="header">Capture Document Using Webcam 📷</div>', unsafe_allow_html=True)

    webrtc_ctx = webrtc_streamer(
        key="document-scanner",
        video_processor_factory=DocumentScanner,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )

    if webrtc_ctx.video_processor:
        if webrtc_ctx.video_processor.result_img is not None:
            scanned_img = webrtc_ctx.video_processor.result_img
            _, buffer = cv.imencode('.png', scanned_img)
            byte_data = buffer.tobytes()
            st.image(cv.cvtColor(scanned_img, cv.COLOR_BGR2RGB), caption="Scanned Document", use_column_width=True)
            st.download_button("Download Scanned Image", byte_data, "scanned_image.png", "image/png", key="webcam_download")
        else:
            st.info("Align the document correctly in front of the camera.")

with col2:
    st.markdown('<div class="header">Upload an Image 🖼</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        img = cv.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv.IMREAD_COLOR)
        processed_img = pre_process_image(img)
        contours = find_document_contours(processed_img)

        if contours is not None:
            scanned_img = warp_document(img, contours)
            col1_upload, col2_upload = st.columns(2)

            with col1_upload:
                st.image(cv.cvtColor(scanned_img, cv.COLOR_BGR2RGB), caption="Scanned Document", use_column_width=True)
                _, buffer = cv.imencode('.png', scanned_img)
                byte_data = buffer.tobytes()
                st.download_button("Download Scanned Image", byte_data, "scanned_image.png", "image/png", key="upload_download")

            with col2_upload:
                st.image(cv.cvtColor(img, cv.COLOR_BGR2RGB), caption="Original Image", use_column_width=True)
            st.success("Document processed successfully!")
        else:
            st.warning("No document detected in the uploaded image.")
