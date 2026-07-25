import streamlit as st
import cv2
import time
from modules.pose_detector import PoseDetector

# ---------------------------------------------------------
# Page Configuration & Aesthetics
# ---------------------------------------------------------
st.set_page_config(
    page_title="AiFitness - Pose Estimation",
    page_icon="🏋️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a professional, premium dark-themed fitness UI
st.markdown("""
<style>
    /* Import modern Outfit font */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    /* Apply font across elements */
    .stApp {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Title Area Design */
    .title-container {
        padding: 1.5rem 0rem;
        margin-bottom: 1rem;
    }
    .main-title {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #8B5CF6 0%, #EC4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        letter-spacing: -1px;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #9CA3AF;
        margin-top: 0.25rem;
        font-weight: 300;
    }
    
    /* Live Status Badge styles */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-bottom: 1.5rem;
    }
    .status-active {
        background-color: rgba(16, 185, 129, 0.12);
        color: #10B981;
        border: 1px solid rgba(16, 185, 129, 0.25);
    }
    .status-active .pulse-dot {
        width: 8px;
        height: 8px;
        background-color: #10B981;
        border-radius: 50%;
        box-shadow: 0 0 8px #10B981;
        animation: pulse 1.5s infinite alternate;
    }
    .status-inactive {
        background-color: rgba(239, 68, 68, 0.12);
        color: #EF4444;
        border: 1px solid rgba(239, 68, 68, 0.25);
    }
    .status-inactive .static-dot {
        width: 8px;
        height: 8px;
        background-color: #EF4444;
        border-radius: 50%;
    }
    
    /* Metrics display */
    .metric-box {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #F3F4F6;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #9CA3AF;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Information panel styling */
    .info-card {
        background: rgba(139, 92, 246, 0.05);
        border: 1px solid rgba(139, 92, 246, 0.15);
        border-radius: 12px;
        padding: 1.2rem;
        margin-top: 1rem;
    }
    .info-title {
        font-weight: 600;
        color: #C084FC;
        margin-bottom: 0.5rem;
    }
    .info-text {
        font-size: 0.9rem;
        color: #D1D5DB;
        line-height: 1.5;
    }
    
    @keyframes pulse {
        0% { transform: scale(0.9); opacity: 0.5; }
        100% { transform: scale(1.25); opacity: 1; }
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# UI Header
# ---------------------------------------------------------
st.markdown("""
<div class="title-container">
    <h1 class="main-title">AiFitness</h1>
    <p class="sub-title">Real-Time Pose Landmarks & Skeletal Stream</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Sidebar Configuration
# ---------------------------------------------------------
st.sidebar.markdown("### ⚙️ Video Settings")

# Checkbox to start/stop the feed
run_feed = st.sidebar.toggle("Start Live Feed", value=True, help="Toggle to open or close the webcam stream.")

# Select camera index
camera_index = st.sidebar.selectbox(
    "Select Camera Source",
    options=[1, 0, 2],
    format_func=lambda x: f"Webcam (Index {x})",
    help="Select the hardware index of the webcam to use."
)

# Pose overlay controls
st.sidebar.markdown("### 🏃 Pose Estimation")
enable_pose = st.sidebar.toggle("Show Skeleton Overlay", value=True, help="Toggle drawing 33 pose skeletal landmarks on stream.")

# Advanced Pose configuration
with st.sidebar.expander("Pose Parameters", expanded=False):
    model_complexity = st.selectbox(
        "Model Complexity",
        options=[0, 1, 2],
        index=1,
        format_func=lambda x: {0: "0 - Lite (Fast)", 1: "1 - Full (Balanced)", 2: "2 - Heavy (Accurate)"}[x],
        help="Higher complexity gives better accuracy but requires more CPU/GPU processing."
    )
    min_det_conf = st.slider("Min Detection Confidence", 0.0, 1.0, 0.5, 0.05)
    min_track_conf = st.slider("Min Tracking Confidence", 0.0, 1.0, 0.5, 0.05)
    min_vis_conf = st.slider("Min Visibility Threshold", 0.0, 1.0, 0.5, 0.05, help="Only draw landmarks/connections if their visibility score is above this value.")

# Visual frame controls
st.sidebar.markdown("### 🎨 Feed Adjustments")
brightness = st.sidebar.slider("Brightness Offset", min_value=-100, max_value=100, value=0, step=5)
contrast = st.sidebar.slider("Contrast Multiplier", min_value=0.5, max_value=2.0, value=1.0, step=0.1)

# ---------------------------------------------------------
# Main Workspace Layout
# ---------------------------------------------------------
col1, col2 = st.columns([2, 1], gap="large")

with col1:
    # Webcam feed display placeholder
    feed_placeholder = st.empty()

with col2:
    # Stats and instructions dashboard
    st.markdown("### 📈 Live Stats")
    
    status_placeholder = st.empty()
    fps_placeholder = st.empty()
    resolution_placeholder = st.empty()
    
    st.markdown("""
    <div class="info-card">
        <div class="info-title">💡 Pose Tracking Tips</div>
        <div class="info-text">
            • <strong>Whole Body:</strong> Ensure all 33 landmark points (including hands, knees, and feet) are visible in the frame.<br/><br/>
            • <strong>Stability:</strong> Use a stable surface or tripod. Shaky video quality decreases tracking accuracy.<br/><br/>
            • <strong>Clothing:</strong> Form-fitting clothes help the Pose model locate joints more reliably.
        </div>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------
# Core Video Capture & Processing Logic
# ---------------------------------------------------------
if run_feed:
    feed_placeholder.info("Initializing webcam capture and Pose model... Please wait.")
    
    # Initialize PoseDetector with configuration options selected from Sidebar
    detector = PoseDetector(
        model_complexity=model_complexity,
        min_detection_confidence=min_det_conf,
        min_tracking_confidence=min_track_conf
    )
    
    # Attempt to open the camera (with CAP_DSHOW on Windows to speed up startup)
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    
    # Fallback to standard capture if CAP_DSHOW fails or hangs
    if not cap.isOpened():
        cap = cv2.VideoCapture(camera_index)
        
    if not cap.isOpened():
        feed_placeholder.error(
            f"Unable to access camera at index {camera_index}.\n\n"
            "Troubleshooting Steps:\n"
            "1. Try changing the camera index in the sidebar.\n"
            "2. Verify that no other application (like Zoom or Teams) is using the webcam.\n"
            "3. Ensure the webcam is connected and drivers are active."
        )
        status_placeholder.markdown("""
        <div class="status-badge status-inactive">
            <div class="static-dot"></div> Connection Error
        </div>
        """, unsafe_allow_html=True)
    else:
        status_placeholder.markdown("""
        <div class="status-badge status-active">
            <div class="pulse-dot"></div> Live Stream Active
        </div>
        """, unsafe_allow_html=True)
        
        # Read resolution once to show in stats
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        resolution_placeholder.markdown(f"""
        <div class="metric-box">
            <div class="metric-value">{width} × {height}</div>
            <div class="metric-label">Resolution (px)</div>
        </div>
        """, unsafe_allow_html=True)
        
        prev_time = 0
        
        try:
            while run_feed:
                ret, frame = cap.read()
                if not ret:
                    st.error("Lost connection to the webcam stream.")
                    break
                
                # Apply real-time brightness and contrast offsets if modified
                if brightness != 0 or contrast != 1.0:
                    frame = cv2.convertScaleAbs(frame, alpha=contrast, beta=brightness)
                
                # Convert BGR (OpenCV default) to RGB (Streamlit & MediaPipe requirement)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Run pose detection if enabled
                if enable_pose:
                    results = detector.find_pose(frame_rgb)
                    # Draw skeletal overlay landmarks onto the RGB frame
                    frame_rgb = detector.draw_skeleton(frame_rgb, results, visibility_threshold=min_vis_conf)
                
                # Render the final processed frame in the streamlit image widget
                feed_placeholder.image(frame_rgb, width="stretch")
                
                # Calculate Frame Rate (FPS)
                current_time = time.time()
                if prev_time > 0:
                    fps = 1.0 / (current_time - prev_time)
                    fps_placeholder.markdown(f"""
                    <div class="metric-box">
                        <div class="metric-value">{fps:.1f}</div>
                        <div class="metric-label">Frames Per Second</div>
                    </div>
                    """, unsafe_allow_html=True)
                prev_time = current_time
                
                # A very short sleep to yield CPU cycles for UI rendering
                time.sleep(0.01)
                
        except Exception as e:
            st.exception(e)
        finally:
            # ALWAYS release the capture device to avoid lockups on next run
            cap.release()
            
            # Clean up stats UI when stopped
            status_placeholder.markdown("""
            <div class="status-badge status-inactive">
                <div class="static-dot"></div> Stream Disconnected
            </div>
            """, unsafe_allow_html=True)
            fps_placeholder.empty()
            resolution_placeholder.empty()
            feed_placeholder.warning("Webcam stream stopped.")
            
else:
    # Offline state UI
    status_placeholder.markdown("""
    <div class="status-badge status-inactive">
        <div class="static-dot"></div> Stream Offline
    </div>
    """, unsafe_allow_html=True)
    
    feed_placeholder.markdown("""
    <div style="border: 2px dashed rgba(255, 255, 255, 0.1); border-radius: 16px; padding: 5rem 2rem; text-align: center; background: rgba(255,255,255,0.01);">
        <h3 style="color: #9CA3AF; margin-bottom: 0.5rem; font-weight: 600;">Feed Offline</h3>
        <p style="color: #6B7280; font-size: 0.95rem; margin-bottom: 1.5rem;">The webcam stream is currently disabled. Toggle the live feed in the sidebar to start streaming.</p>
    </div>
    """, unsafe_allow_html=True)