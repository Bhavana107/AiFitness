import streamlit as st

def get_color_class(value: float, thresholds: tuple) -> str:
    """
    Returns a CSS style color based on value and thresholds.
    thresholds = (yellow_limit, green_limit)
    """
    # For visibilities/confidences, higher is better
    val_min, val_max = thresholds
    if value >= val_max:
        return "#10B981"  # Stable Green
    elif value >= val_min:
        return "#F59E0B"  # Warning Yellow
    else:
        return "#EF4444"  # Unstable Red

def render_debug_panel(container, fps: float, tracker, visibilities: dict, min_vis_conf: float):
    """
    Renders a live, color-coded debug panel inside the provided Streamlit container.
    
    Args:
        container: Streamlit placeholder (st.empty()).
        fps (float): Current Frames Per Second.
        tracker: The active SquatTracker instance.
        visibilities (dict): Dictionary of landmark ID -> visibility score.
        min_vis_conf (float): The minimum visibility confidence set in UI.
    """
    if tracker is None:
        container.empty()
        return

    # 1. Calculate visibilities for joint groups
    # Default to 0.0 if not detected
    def get_group_vis(indices):
        if not visibilities:
            return 0.0
        vals = [visibilities.get(i, 0.0) for i in indices]
        return sum(vals) / len(vals)

    l_knee_vis = get_group_vis([23, 25, 27])
    r_knee_vis = get_group_vis([24, 26, 28])
    avg_knee_vis = (l_knee_vis + r_knee_vis) / 2.0
    
    l_hip_vis = get_group_vis([11, 23, 25])
    r_hip_vis = get_group_vis([12, 24, 26])
    avg_hip_vis = (l_hip_vis + r_hip_vis) / 2.0
    
    l_elbow_vis = get_group_vis([11, 13, 15])
    r_elbow_vis = get_group_vis([12, 14, 16])
    
    overall_vis = get_group_vis(range(33)) if visibilities else 0.0

    # 2. Determine colors for stats
    fps_color = get_color_class(fps, (12.0, 24.0))
    l_knee_color = get_color_class(l_knee_vis, (min_vis_conf, min_vis_conf + 0.15))
    r_knee_color = get_color_class(r_knee_vis, (min_vis_conf, min_vis_conf + 0.15))
    avg_knee_color = get_color_class(avg_knee_vis, (min_vis_conf, min_vis_conf + 0.15))
    
    l_hip_color = get_color_class(l_hip_vis, (min_vis_conf, min_vis_conf + 0.15))
    r_hip_color = get_color_class(r_hip_vis, (min_vis_conf, min_vis_conf + 0.15))
    avg_hip_color = get_color_class(avg_hip_vis, (min_vis_conf, min_vis_conf + 0.15))
    
    l_elbow_color = get_color_class(l_elbow_vis, (min_vis_conf, min_vis_conf + 0.15))
    r_elbow_color = get_color_class(r_elbow_vis, (min_vis_conf, min_vis_conf + 0.15))
    
    pose_conf_color = get_color_class(overall_vis, (0.45, 0.70))

    # 3. Determine Rep Validity and State colors
    state_colors = {
        "STANDING": "#10B981",    # Green (Ready)
        "DESCENDING": "#FBBF24",  # Yellow
        "BOTTOM": "#10B981",      # Green (Depth hit)
        "ASCENDING": "#10B981"    # Green
    }
    state_color = state_colors.get(tracker.state, "#9CA3AF")
    
    dir_colors = {
        "STATIONARY": "#9CA3AF",
        "DESCENDING": "#FBBF24",
        "ASCENDING": "#10B981"
    }
    dir_color = dir_colors.get(tracker.direction, "#9CA3AF")

    # Rep Validity Logic
    if overall_vis < min_vis_conf:
        validity_text = "INVALID (Pose untracked)"
        validity_color = "#EF4444" # Red
    else:
        if tracker.state == "STANDING":
            validity_text = "VALID (Ready)"
            validity_color = "#10B981" # Green
        elif tracker.state == "DESCENDING":
            if tracker.smoothed_avg_knee > tracker.angle_bottom:
                validity_text = "INVALID (Depth insufficient)"
                validity_color = "#F59E0B" # Yellow
            else:
                validity_text = "VALID (Target depth hit!)"
                validity_color = "#10B981" # Green
        elif tracker.state in ["BOTTOM", "ASCENDING"]:
            validity_text = "VALID (Good depth!)"
            validity_color = "#10B981" # Green
        else:
            validity_text = "UNKNOWN"
            validity_color = "#9CA3AF"

    # Formatted Strings
    l_knee_str = f"{int(tracker.smoothed_left_knee)}°" if l_knee_vis >= min_vis_conf else "--"
    r_knee_str = f"{int(tracker.smoothed_right_knee)}°" if r_knee_vis >= min_vis_conf else "--"
    avg_knee_str = f"{int(tracker.smoothed_avg_knee)}°" if avg_knee_vis >= min_vis_conf else "--"
    
    l_hip_str = f"{int(tracker.smoothed_left_hip)}°" if l_hip_vis >= min_vis_conf else "--"
    r_hip_str = f"{int(tracker.smoothed_right_hip)}°" if r_hip_vis >= min_vis_conf else "--"
    avg_hip_str = f"{int(tracker.smoothed_avg_hip)}°" if avg_hip_vis >= min_vis_conf else "--"
    
    # Elbow angles are calculated directly from smoothers if they exist
    # Let's check if the tracker holds them. Since the request mentions:
    # "smoothed_left_knee, smoothed_right_knee, etc. are initialized correctly",
    # and elbows are not in the main FSM tracker (since they don't drive squats),
    # we can compute or read them dynamically.
    # In draw_skeleton we draw them. In render_debug_panel, let's display them.
    # To display them here, we will fetch their values.
    # If the user has calculated the elbow angles in app.py, we can pass them in or compute them here.
    # Let's pass them as parameters: l_elbow, r_elbow.
    l_elbow_angle = tracker.smoothed_left_elbow if hasattr(tracker, "smoothed_left_elbow") else None
    r_elbow_angle = tracker.smoothed_right_elbow if hasattr(tracker, "smoothed_right_elbow") else None
    
    l_elbow_str = f"{int(l_elbow_angle)}°" if (l_elbow_angle is not None and l_elbow_vis >= min_vis_conf) else "--"
    r_elbow_str = f"{int(r_elbow_angle)}°" if (r_elbow_angle is not None and r_elbow_vis >= min_vis_conf) else "--"

    # Render HUD Card Markdown
    container.markdown(f"""
    <div style="background: rgba(31, 41, 55, 0.4); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px; padding: 1.2rem; margin-top: 1.5rem;">
        <h4 style="color: #EC4899; margin-top: 0; margin-bottom: 1rem; font-size: 1.1rem; border-bottom: 1px solid rgba(255,255,255,0.05); padding-bottom: 0.5rem;">🛠️ Pose Debug Panel</h4>
        
        <table style="width: 100%; border-collapse: collapse; font-family: monospace; font-size: 0.9rem;">
            <tr style="border-bottom: 1px solid rgba(255,255,255,0.03);">
                <td style="color: #9CA3AF; padding: 6px 0;">Frames Per Second (FPS)</td>
                <td style="text-align: right; font-weight: bold; color: {fps_color};">{fps:.1f}</td>
            </tr>
            <tr style="border-bottom: 1px solid rgba(255,255,255,0.03);">
                <td style="color: #9CA3AF; padding: 6px 0;">Pose Detection Conf</td>
                <td style="text-align: right; font-weight: bold; color: {pose_conf_color};">{overall_vis * 100:.1f}%</td>
            </tr>
            <tr style="border-bottom: 1px solid rgba(255,255,255,0.03);">
                <td style="color: #9CA3AF; padding: 6px 0;">Rep Validity</td>
                <td style="text-align: right; font-weight: bold; color: {validity_color};">{validity_text}</td>
            </tr>
            <tr style="border-bottom: 1px solid rgba(255,255,255,0.03);">
                <td style="color: #9CA3AF; padding: 6px 0;">Rep Count</td>
                <td style="text-align: right; font-weight: bold; color: #10B981;">{tracker.rep_count}</td>
            </tr>
            <tr style="border-bottom: 1px solid rgba(255,255,255,0.03);">
                <td style="color: #9CA3AF; padding: 6px 0;">Movement State</td>
                <td style="text-align: right; font-weight: bold; color: {state_color};">{tracker.state}</td>
            </tr>
            <tr style="border-bottom: 1px solid rgba(255,255,255,0.03);">
                <td style="color: #9CA3AF; padding: 6px 0;">Movement Direction</td>
                <td style="text-align: right; font-weight: bold; color: {dir_color};">{tracker.direction}</td>
            </tr>
            <tr style="border-bottom: 1px solid rgba(255,255,255,0.03);">
                <td style="color: #9CA3AF; padding: 6px 0;">Left Knee Angle</td>
                <td style="text-align: right; font-weight: bold; color: {l_knee_color};">{l_knee_str}</td>
            </tr>
            <tr style="border-bottom: 1px solid rgba(255,255,255,0.03);">
                <td style="color: #9CA3AF; padding: 6px 0;">Right Knee Angle</td>
                <td style="text-align: right; font-weight: bold; color: {r_knee_color};">{r_knee_str}</td>
            </tr>
            <tr style="border-bottom: 1px solid rgba(255,255,255,0.03);">
                <td style="color: #9CA3AF; padding: 6px 0;">Average Knee Angle</td>
                <td style="text-align: right; font-weight: bold; color: {avg_knee_color};">{avg_knee_str}</td>
            </tr>
            <tr style="border-bottom: 1px solid rgba(255,255,255,0.03);">
                <td style="color: #9CA3AF; padding: 6px 0;">Left Hip Angle</td>
                <td style="text-align: right; font-weight: bold; color: {l_hip_color};">{l_hip_str}</td>
            </tr>
            <tr style="border-bottom: 1px solid rgba(255,255,255,0.03);">
                <td style="color: #9CA3AF; padding: 6px 0;">Right Hip Angle</td>
                <td style="text-align: right; font-weight: bold; color: {r_hip_color};">{r_hip_str}</td>
            </tr>
            <tr style="border-bottom: 1px solid rgba(255,255,255,0.03);">
                <td style="color: #9CA3AF; padding: 6px 0;">Average Hip Angle</td>
                <td style="text-align: right; font-weight: bold; color: {avg_hip_color};">{avg_hip_str}</td>
            </tr>
            <tr style="border-bottom: 1px solid rgba(255,255,255,0.03);">
                <td style="color: #9CA3AF; padding: 6px 0;">Left Elbow Angle</td>
                <td style="text-align: right; font-weight: bold; color: {l_elbow_color};">{l_elbow_str}</td>
            </tr>
            <tr>
                <td style="color: #9CA3AF; padding: 6px 0;">Right Elbow Angle</td>
                <td style="text-align: right; font-weight: bold; color: {r_elbow_color};">{r_elbow_str}</td>
            </tr>
        </table>
    </div>
    """, unsafe_allow_html=True)
