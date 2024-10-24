# SNAPnFIX: Document Scanner and Fixer

## Overview

Welcome to **SNAPnFIX** — an interactive and user-friendly document scanning app built using Streamlit! SNAPnFIX allows users to scan documents either through a **webcam** or by **uploading an image** from their local machine. This app is designed for ease of use, with simple controls, real-time document detection, and an enhanced document sharpening feature for clear and crisp results.

### Key Features

1. **Document Capture via Webcam**  
   The app offers a live camera feed to capture documents using your **webcam**. After detecting the document, it applies edge detection and warps the image to provide a clean, flattened scan of the document.

2. **Upload an Image for Scanning**  
   Users can also upload an image (in `.jpg`, `.jpeg`, or `.png` format) from their local machine, and SNAPnFIX will process the image to detect and scan the document.

3. **Real-Time Document Detection**  
   The app uses image preprocessing techniques like grayscale conversion, Gaussian blurring, and Canny edge detection to identify document boundaries in real time. Once detected, it applies a **perspective transformation** to create a properly aligned scan.

4. **Interactive Buttons**  
   - **Capture Document**: Start the webcam capture process to scan a document.
   - **Upload an Image**: Choose and upload an image from your local machine for scanning.
   - **Download Button**: Once the document is processed, users can download the scanned image directly from the app in **PNG** format.

5. **Downloadable Scanned Image**  
   After successfully scanning a document (whether from webcam capture or file upload), users can instantly download the processed image with the click of a button.

6. **Sharpening for Clarity**  
   To enhance text readability, a sharpening filter is applied to the scanned document, ensuring that the text and details are as clear as possible.

### How SNAPnFIX Works

1. **Preprocessing the Image**  
   The app converts the captured or uploaded image to grayscale, applies Gaussian blur to smooth out the noise, and then uses Canny edge detection to find the document's edges.

2. **Detect and Draw Contours**  
   After preprocessing, the app identifies the contours of the document. If a four-corner contour is detected, it proceeds to warp the document into a flat, rectangular scan.

3. **Reordering Points for Perspective Transformation**  
   Once the corners are detected, they are reordered to ensure that the document is properly oriented. This allows the app to perform a perspective transformation, resulting in a clear, aligned document scan.

4. **Perspective Transformation for Warping the Document**  
   The app warps the document, using the reordered corner points, into a rectangular output that simulates a scan of the document.

### How to Use SNAPnFIX Locally

Follow these simple steps to get SNAPnFIX up and running on your local machine:

1. **Clone the Repository**  
   Start by cloning the SNAPnFIX repository (you'll need to have Git installed):
   ```bash
   git clone <https://github.com/GeeteshPaidi/SNAPnFIX-Document-Scanner.git>
2. **Navigate to the Project Directory**    
   Move into the directory where the project files are stored:
   ```bash
   cd SNAPnFIX
3. **Install the Dependencies**   
   Install the required Python packages listed in the `requirements.txt` file:
   ```bash
   pip install -r requirements.txt
4. **Run the Application**   
   Once everything is set up, you can run the Streamlit app:
   ```bash
   streamlit run app.py
5. **Access the App**     
   After running the command, your browser should automatically open the SNAPnFIX app. If not, visit the local URL provided in the terminal (usually `http://localhost:8501`).

### How to Use the App
+ **Using the Webcam:**  
  Click on the Start Document Capture button under the webcam section. The app will launch the webcam, detect the document, and display the scanned version. You can download the scanned image after it’s processed.

+ **Uploading an Image:**   
  Under the Upload an Image section, choose an image from your computer. The app will process the image, detect the document, and allow you to download the scanned version.
 
+ **Downloading the Scanned Document:**     
  Once a document is captured or processed, click the Download Scanned Image button to save the scanned document in PNG format.

### Planned Cloud Deployment
I'm currently working on deploying SNAPnFIX to the cloud. Once deployed, users will be able to access the app directly through a web link, making it easier for everyone to scan documents without needing to install anything locally. Stay tuned for updates on the deployment progress!




