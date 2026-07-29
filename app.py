import streamlit as st
import cv2
import time
from modules.pose_detector import PoseDetector
from modules.rep_counter import SquatTracker
from modules.geometry import calculate_angle
from modules.debug_panel import render_debug_panel
from modules.database import init_db, log_workout, get_workout_history, clear_workout_history

# Initialize database
init_db()

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

# Initialize session state for squat tracker (force overwrite if old class is cached)
if "squat_counter" not in st.session_state or not isinstance(st.session_state.squat_counter, SquatTracker):
    st.session_state.squat_counter = SquatTracker()

if "session_start_time" not in st.session_state:
    st.session_state.session_start_time = None
if "fps_history" not in st.session_state:
    st.session_state.fps_history = []

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

# Initialize session state for debug panel visibility toggle
if "show_debug" not in st.session_state:
    st.session_state.show_debug = False

# Reset and Save controls
col_reset, col_save = st.sidebar.columns(2)
with col_reset:
    if st.button("Reset Reps", use_container_width=True):
        st.session_state.squat_counter.reset()
        st.session_state.session_start_time = time.time() if run_feed else None
        st.session_state.fps_history = []
        st.toast("Rep counter reset!")
with col_save:
    if st.button("Save Session", use_container_width=True):
        tracker = st.session_state.squat_counter
        if tracker.rep_count > 0 or (st.session_state.session_start_time is not None and (time.time() - st.session_state.session_start_time) > 2.0):
            duration = 0.0
            if st.session_state.session_start_time is not None:
                duration = time.time() - st.session_state.session_start_time
            
            avg_fps = 0.0
            if st.session_state.fps_history:
                avg_fps = sum(st.session_state.fps_history) / len(st.session_state.fps_history)
                
            success = log_workout(
                exercise="Squats",
                reps=tracker.rep_count,
                duration=duration,
                avg_fps=avg_fps
            )
            if success:
                st.success(f"Session saved: {tracker.rep_count} reps!")
                tracker.reset()
                st.session_state.session_start_time = time.time() if run_feed else None
                st.session_state.fps_history = []
                st.rerun()
            else:
                st.warning("Nothing to save yet.")
        else:
            st.warning("No reps completed to save.")

debug_btn_label = "Hide Debug Panel" if st.session_state.show_debug else "Show Debug Panel"
if st.sidebar.button(debug_btn_label, width="stretch"):
    st.session_state.show_debug = not st.session_state.show_debug

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
    rep_placeholder = st.empty()
    phase_placeholder = st.empty()
    knee_angle_placeholder = st.empty()
    fps_placeholder = st.empty()
    resolution_placeholder = st.empty()
    
    # Placeholder for the dynamic Pose Debug Panel
    debug_placeholder = st.empty()
    
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
        
        if st.session_state.session_start_time is None:
            st.session_state.session_start_time = time.time()
            st.session_state.fps_history = []
            
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
                
                # Initialize variables for debug panel and calculations
                landmarks = None
                
                # Run pose detection if enabled
                if enable_pose:
                    results = detector.find_pose(frame_rgb)
                    
                    # ---------------------------------------------------------
                    # Squat Tracker Processing
                    # ---------------------------------------------------------
                    landmarks = detector.get_landmarks_list(results, width, height)
                    if landmarks:
                        # Extract joints
                        l_shoulder = landmarks[11]
                        r_shoulder = landmarks[12]
                        l_hip = landmarks[23]
                        r_hip = landmarks[24]
                        l_knee = landmarks[25]
                        r_knee = landmarks[26]
                        l_ankle = landmarks[27]
                        r_ankle = landmarks[28]
                        
                        # Check joint visibilities
                        l_shoulder_valid = l_shoulder['visibility'] >= min_vis_conf
                        r_shoulder_valid = r_shoulder['visibility'] >= min_vis_conf
                        l_hip_valid = l_hip['visibility'] >= min_vis_conf
                        r_hip_valid = r_hip['visibility'] >= min_vis_conf
                        l_knee_valid = l_knee['visibility'] >= min_vis_conf
                        r_knee_valid = r_knee['visibility'] >= min_vis_conf
                        l_ankle_valid = l_ankle['visibility'] >= min_vis_conf
                        r_ankle_valid = r_ankle['visibility'] >= min_vis_conf
                        
                        # Torso vertical coordinates (normalized Y)
                        shoulder_y_norm = None
                        if l_shoulder_valid and r_shoulder_valid:
                            shoulder_y_norm = ((l_shoulder['y'] + r_shoulder['y']) / 2.0) / height
                        elif l_shoulder_valid:
                            shoulder_y_norm = l_shoulder['y'] / height
                        elif r_shoulder_valid:
                            shoulder_y_norm = r_shoulder['y'] / height
                            
                        hip_y_norm = None
                        if l_hip_valid and r_hip_valid:
                            hip_y_norm = ((l_hip['y'] + r_hip['y']) / 2.0) / height
                        elif l_hip_valid:
                            hip_y_norm = l_hip['y'] / height
                        elif r_hip_valid:
                            hip_y_norm = r_hip['y'] / height
                            
                        # Joint angles calculation
                        l_knee_angle = None
                        if l_hip_valid and l_knee_valid and l_ankle_valid:
                            l_knee_angle = calculate_angle(
                                (l_hip['x'], l_hip['y']),
                                (l_knee['x'], l_knee['y']),
                                (l_ankle['x'], l_ankle['y'])
                            )
                            
                        r_knee_angle = None
                        if r_hip_valid and r_knee_valid and r_ankle_valid:
                            r_knee_angle = calculate_angle(
                                (r_hip['x'], r_hip['y']),
                                (r_knee['x'], r_knee['y']),
                                (r_ankle['x'], r_ankle['y'])
                            )
                            
                        l_hip_angle = None
                        if l_shoulder_valid and l_hip_valid and l_knee_valid:
                            l_hip_angle = calculate_angle(
                                (l_shoulder['x'], l_shoulder['y']),
                                (l_hip['x'], l_hip['y']),
                                (l_knee['x'], l_knee['y'])
                            )
                            
                        r_hip_angle = None
                        if r_shoulder_valid and r_hip_valid and r_knee_valid:
                            r_hip_angle = calculate_angle(
                                (r_shoulder['x'], r_shoulder['y']),
                                (r_hip['x'], r_hip['y']),
                                (r_knee['x'], r_knee['y'])
                            )
                            
                        l_elbow_angle = None
                        if l_shoulder_valid and landmarks[13]['visibility'] >= min_vis_conf and landmarks[15]['visibility'] >= min_vis_conf:
                            l_elbow_angle = calculate_angle(
                                (l_shoulder['x'], l_shoulder['y']),
                                (landmarks[13]['x'], landmarks[13]['y']),
                                (landmarks[15]['x'], landmarks[15]['y'])
                            )
                            
                        r_elbow_angle = None
                        if r_shoulder_valid and landmarks[14]['visibility'] >= min_vis_conf and landmarks[16]['visibility'] >= min_vis_conf:
                            r_elbow_angle = calculate_angle(
                                (r_shoulder['x'], r_shoulder['y']),
                                (landmarks[14]['x'], landmarks[14]['y']),
                                (landmarks[16]['x'], landmarks[16]['y'])
                            )
                            
                        # Update the tracker with both knees, hips, shoulders, and elbows
                        st.session_state.squat_counter.update(
                            l_knee=l_knee_angle,
                            r_knee=r_knee_angle,
                            l_hip=l_hip_angle,
                            r_hip=r_hip_angle,
                            shoulder_y=shoulder_y_norm,
                            hip_y=hip_y_norm,
                            l_elbow=l_elbow_angle,
                            r_elbow=r_elbow_angle
                        )
                        
                    # Draw skeletal overlay landmarks and HUD overlay onto the RGB frame
                    tracker = st.session_state.squat_counter
                    frame_rgb = detector.draw_skeleton(
                        frame_rgb, 
                        results, 
                        visibility_threshold=min_vis_conf,
                        state=tracker.state,
                        direction=tracker.direction,
                        reps=tracker.rep_count,
                        l_knee=tracker.smoothed_left_knee,
                        r_knee=tracker.smoothed_right_knee,
                        avg_knee=tracker.smoothed_avg_knee,
                        avg_hip=tracker.smoothed_avg_hip
                    )
                else:
                    # Draw empty HUD if pose detection is disabled
                    tracker = st.session_state.squat_counter
                    frame_rgb = detector.draw_skeleton(
                        frame_rgb,
                        None,
                        visibility_threshold=min_vis_conf,
                        state=tracker.state,
                        direction=tracker.direction,
                        reps=tracker.rep_count
                    )

                # Render the final processed frame in the streamlit image widget
                feed_placeholder.image(frame_rgb, width="stretch")
                
                # Update rep counter and phase displays dynamically
                counter = st.session_state.squat_counter
                
                rep_placeholder.markdown(f"""
                <div class="metric-box" style="background: rgba(139, 92, 246, 0.08); border-color: rgba(139, 92, 246, 0.25);">
                    <div class="metric-value" style="color: #C084FC; font-size: 2.8rem;">{counter.rep_count}</div>
                    <div class="metric-label">Rep Count</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Determine theme color for phase
                state_colors = {
                    "STANDING": "#9CA3AF",      # Gray
                    "DESCENDING": "#FBBF24",    # Amber/Yellow
                    "BOTTOM": "#EF4444",        # Red
                    "ASCENDING": "#10B981"      # Emerald/Green
                }
                phase_color = state_colors.get(counter.state, "#9CA3AF")
                
                phase_placeholder.markdown(f"""
                <div class="metric-box" style="border-left: 4px solid {phase_color};">
                    <div class="metric-value" style="color: {phase_color}; font-size: 1.6rem;">{counter.state}</div>
                    <div class="metric-label">Squat Phase ({counter.direction})</div>
                </div>
                """, unsafe_allow_html=True)
                
                angle_str = f"{int(counter.smoothed_avg_knee)}°" if counter.smoothed_avg_knee > 0 else "--"
                knee_angle_placeholder.markdown(f"""
                <div class="metric-box">
                    <div class="metric-value">{angle_str}</div>
                    <div class="metric-label">Avg Knee Angle (Smoothed)</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Calculate Frame Rate (FPS)
                current_time = time.time()
                fps = 0.0
                if prev_time > 0:
                    fps = 1.0 / (current_time - prev_time)
                    st.session_state.fps_history.append(fps)
                    fps_placeholder.markdown(f"""
                    <div class="metric-box">
                        <div class="metric-value">{fps:.1f}</div>
                        <div class="metric-label">Frames Per Second</div>
                    </div>
                    """, unsafe_allow_html=True)
                prev_time = current_time
                
                # Render live Debug Panel if enabled
                if st.session_state.show_debug:
                    visibilities = {}
                    if landmarks:
                        visibilities = {lm['id']: lm['visibility'] for lm in landmarks}
                    
                    render_debug_panel(
                        container=debug_placeholder,
                        fps=fps,
                        tracker=st.session_state.squat_counter,
                        visibilities=visibilities,
                        min_vis_conf=min_vis_conf
                    )
                else:
                    debug_placeholder.empty()
                
                # A very short sleep to yield CPU cycles for UI rendering
                time.sleep(0.01)
                
        except Exception as e:
            st.exception(e)
        finally:
            # ALWAYS release the capture device to avoid lockups on next run
            cap.release()
            
            # Auto-save workout if stream is stopped and reps are > 0
            if st.session_state.session_start_time is not None:
                tracker = st.session_state.squat_counter
                if tracker.rep_count > 0:
                    duration = time.time() - st.session_state.session_start_time
                    avg_fps = 0.0
                    if st.session_state.fps_history:
                        avg_fps = sum(st.session_state.fps_history) / len(st.session_state.fps_history)
                    
                    log_workout(
                        exercise="Squats",
                        reps=tracker.rep_count,
                        duration=duration,
                        avg_fps=avg_fps
                    )
                    st.toast(f"Auto-saved session: {tracker.rep_count} squats!")
                
                # Clear session state variables
                st.session_state.session_start_time = None
                st.session_state.fps_history = []

            # Clean up stats UI when stopped
            status_placeholder.markdown("""
            <div class="status-badge status-inactive">
                <div class="static-dot"></div> Stream Disconnected
            </div>
            """, unsafe_allow_html=True)
            rep_placeholder.empty()
            phase_placeholder.empty()
            knee_angle_placeholder.empty()
            debug_placeholder.empty()
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
    rep_placeholder.empty()
    phase_placeholder.empty()
    knee_angle_placeholder.empty()
    debug_placeholder.empty()
    
    feed_placeholder.markdown("""
    <div style="border: 2px dashed rgba(255, 255, 255, 0.1); border-radius: 16px; padding: 5rem 2rem; text-align: center; background: rgba(255,255,255,0.01);">
        <h3 style="color: #9CA3AF; margin-bottom: 0.5rem; font-weight: 600;">Feed Offline</h3>
        <p style="color: #6B7280; font-size: 0.95rem; margin-bottom: 1.5rem;">The webcam stream is currently disabled. Toggle the live feed in the sidebar to start streaming.</p>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------
# Workout History Section
# ---------------------------------------------------------
st.markdown("---")
st.markdown("### 📊 Workout History")

# Retrieve and display history database table
df_history = get_workout_history()

if not df_history.empty:
    st.dataframe(
        df_history, 
        use_container_width=True,
        column_config={
            "Date": st.column_config.TextColumn("Date"),
            "Time": st.column_config.TextColumn("Time"),
            "Exercise": st.column_config.TextColumn("Exercise"),
            "Reps": st.column_config.NumberColumn("Reps", format="%d"),
            "Duration (s)": st.column_config.NumberColumn("Duration (s)", format="%.1f s"),
            "Avg FPS": st.column_config.NumberColumn("Avg FPS", format="%.1f")
        }
    )
    
    # Optional clear history action
    if st.button("Delete Workout History", use_container_width=False):
        clear_workout_history()
        st.toast("Workout history cleared successfully.")
        st.rerun()
else:
    st.info("No workouts recorded yet. Start training to log your first session!")