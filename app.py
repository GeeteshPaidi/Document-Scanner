# importing necessary libraries
import cv2 as cv
import numpy as np
import streamlit as st

# setting up streamlit for some good visuals
st.set_page_config(page_title="SNAPnFIX", page_icon="📄", layout="wide")
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
    .button-area {
        text-align: center;
        margin-top: 20px;
    }
    .highlight {
        background-color: #2a2a2a;
        border-radius: 10px;
        padding: 10px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">📄 SNAPnFIX: Document Scanner and Fixer</div>', unsafe_allow_html=True)

# this can be played with
frame_width = 640
frame_height = 480
contour_area = 0.3 * frame_width * frame_height

# preprocessing the image
def pre_processing(img):
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img_blur = cv.GaussianBlur(img_gray, (3, 3), 1)  # to reduce noice
    img_canny = cv.Canny(img_blur, 30, 150)
    kernel = np.ones((5, 5))
    img_dial = cv.dilate(img_canny, kernel, iterations=3) # to increase the visibility of border
    img_erode = cv.erode(img_dial, kernel, iterations=1)
    return img_erode

# detect and draw contours
def draw_contour(img):
    contours, _ = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv.contourArea(cnt)
        if area > contour_area:
            peri = cv.arcLength(cnt, True)
            approx = cv.approxPolyDP(cnt, 0.02 * peri, True)
            if area > contour_area and len(approx) == 4: # 4 coz box has 4 sides
                return approx
    return np.array([])

# reordering points for perspective transformation
def reorder(my_points):
    my_points = my_points.reshape((4, 2))
    my_points_new = np.zeros((4, 1, 2), np.int32)
    add = my_points.sum(1)
    my_points_new[0] = my_points[np.argmin(add)]  # Top-left
    my_points_new[3] = my_points[np.argmax(add)]  # Bottom-right
    diff = np.diff(my_points, axis=1)
    my_points_new[1] = my_points[np.argmin(diff)]  # Top-right
    my_points_new[2] = my_points[np.argmax(diff)]  # Bottom-left
    return my_points_new

# perspective transform for warping the document
def get_warp(img, captured_img):
    captured_img = reorder(captured_img)
    pts1 = np.float32(captured_img)
    x, y, w, h = cv.boundingRect(captured_img)    # this would make the output image size fit with scanned document
    pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    matrix = cv.getPerspectiveTransform(pts1, pts2)
    img_output = cv.warpPerspective(img, matrix, (w, h))

    # sharpening kernel for text clarity
    sharpening_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    img_output = cv.filter2D(img_output, -1, sharpening_kernel)

    return img_output

# main layout
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown('<div class="header">Capture Document Using Webcam 📷</div>', unsafe_allow_html=True)
    run = st.button('Start Document Capture')

    FRAME_WINDOW = st.image([])

    if run:
        cap = cv.VideoCapture(0)
        cap.set(3, frame_width)
        cap.set(4, frame_height)
        cap.set(10, 150)

        with st.spinner('Capturing...'):
            while run:
                success, img = cap.read()
                if not success:
                    st.error("Failed to capture image")
                    break

                result_img = pre_processing(img)
                final_img = draw_contour(result_img)

                if final_img.size != 0:
                    warp_img = get_warp(img, final_img)
                    FRAME_WINDOW.image(cv.cvtColor(warp_img, cv.COLOR_BGR2RGB), channels="RGB")
                    _, buffer = cv.imencode('.png', warp_img)
                    byte_data = buffer.tobytes()
                    st.image(cv.cvtColor(img, cv.COLOR_BGR2RGB), caption="Original Image", use_column_width=False)
                    st.success("Document detected and captured!")
                    st.download_button("Download Scanned Image", byte_data, "scanned_image.png", "image/png", key="webcam_download")
                    break
                else:
                    FRAME_WINDOW.image(cv.cvtColor(img, cv.COLOR_BGR2RGB), channels="RGB")

        cap.release()

with col2:
    st.markdown('<div class="header">Upload an Image 🖼</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        img = cv.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
        result_img = pre_processing(img)
        final_img = draw_contour(result_img)

        if final_img.size != 0:
            warp_img = get_warp(img, final_img)
            col1_upload, col2_upload = st.columns(2)
            with col1_upload:
                st.image(cv.cvtColor(warp_img, cv.COLOR_BGR2RGB), caption="Scanned Document", use_column_width=True)
                _, buffer = cv.imencode('.png', warp_img)
                byte_data = buffer.tobytes()
                st.download_button("Download Scanned Image", byte_data, "scanned_image.png", "image/png", key="upload_download")

            with col2_upload:
                st.image(cv.cvtColor(img, cv.COLOR_BGR2RGB), caption="Original Image", use_column_width=True)
            st.success("Document processed successfully!")
        else:
            st.warning("No document detected in the uploaded image.")
