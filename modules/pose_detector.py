import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os

class PoseDetector:
    """
    A modular helper class to encapsulate MediaPipe Tasks Pose Landmarker logic.
    Provides methods to initialize the detector, find pose landmarks on RGB frames,
    and render the full skeleton/landmark points along with joint angle overlays.
    """
    # Official complete list of 35 connections (standard 33-pose skeleton connections)
    # matching the official MediaPipe Pose model specification.
    POSE_CONNECTIONS = [
        # Face / Head
        (0, 1), (1, 2), (2, 3), (3, 7),       # Nose -> Left Eye -> Left Ear
        (0, 4), (4, 5), (5, 6), (6, 8),       # Nose -> Right Eye -> Right Ear
        (9, 10),                              # Mouth corners
        # Torso / Hips
        (11, 12),                             # Shoulder to Shoulder
        (11, 23),                             # Left Shoulder to Left Hip
        (12, 24),                             # Right Shoulder to Right Hip
        (23, 24),                             # Hip to Hip
        # Left Upper Extremity (Arm & Hand)
        (11, 13),                             # Left Shoulder to Left Elbow
        (13, 15),                             # Left Elbow to Left Wrist
        (15, 17), (15, 19), (15, 21),         # Left Wrist to Pinky / Index / Thumb
        (17, 19),                             # Left Pinky to Left Index
        # Right Upper Extremity (Arm & Hand)
        (12, 14),                             # Right Shoulder to Right Elbow
        (14, 16),                             # Right Elbow to Right Wrist
        (16, 18), (16, 20), (16, 22),         # Right Wrist to Pinky / Index / Thumb
        (18, 20),                             # Right Pinky to Right Index
        # Left Lower Extremity (Leg & Foot)
        (23, 25),                             # Left Hip to Left Knee
        (25, 27),                             # Left Knee to Left Ankle
        (27, 29),                             # Left Ankle to Left Heel
        (29, 31),                             # Left Heel to Left Foot Index
        (27, 31),                             # Left Ankle to Left Foot Index (foot triangle)
        # Right Lower Extremity (Leg & Foot)
        (24, 26),                             # Right Hip to Right Knee
        (26, 28),                             # Right Knee to Right Ankle
        (28, 30),                             # Right Ankle to Right Heel
        (30, 32),                             # Right Heel to Right Foot Index
        (28, 32)                              # Right Ankle to Right Foot Index (foot triangle)
    ]

    def __init__(
        self,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5
    ):
        # Map model complexity option to local .task file path
        model_map = {
            0: "models/pose_landmarker_lite.task",
            1: "models/pose_landmarker_full.task",
            2: "models/pose_landmarker_heavy.task"
        }
        model_path = model_map.get(model_complexity, "models/pose_landmarker_full.task")
        
        # Verify the model file exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found at: {model_path}. "
                "Ensure models are downloaded correctly in the models/ directory."
            )

        # Configure the MediaPipe Vision task options
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE, # Frame-by-frame image mode
            num_poses=1,
            min_pose_detection_confidence=min_detection_confidence,
            min_pose_presence_confidence=min_tracking_confidence
        )
        
        # Create the pose detector instance
        self.detector = vision.PoseLandmarker.create_from_options(options)

    def find_pose(self, frame_rgb):
        """
        Processes a single RGB frame to detect human poses.
        Note: The frame is passed in full without any cropping or resizing 
        to ensure full body landmark coordinate accuracy.
        
        Args:
            frame_rgb (np.ndarray): An RGB image frame.
            
        Returns:
            PoseLandmarkerResult: The detection results containing landmarks.
        """
        # Convert NumPy RGB frame to MediaPipe Image object
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        
        # Detect pose landmarks on the full frame
        return self.detector.detect(mp_image)

    def draw_skeleton(self, frame_rgb, results, visibility_threshold: float = 0.5):
        """
        Draws the full 33 pose landmarks and connection skeleton on the frame.
        Filters joints and lines based on the visibility threshold to prevent 
        inaccuracies or jitter from occluded limbs (e.g. when lower body is hidden).
        Also calculates and displays joint angles for elbows, knees, and hips.
        
        Args:
            frame_rgb (np.ndarray): An RGB image frame (numpy array).
            results (PoseLandmarkerResult): Pose detection results.
            visibility_threshold (float): Minimum confidence required to draw a point/line.
            
        Returns:
            np.ndarray: The modified RGB image frame with drawn landmarks.
        """
        if not results.pose_landmarks:
            return frame_rgb
        
        h, w, c = frame_rgb.shape
        
        # Import the geometry module internally to keep code clean and modular
        from modules.geometry import calculate_angle
        
        # Visual theme drawing colors matching styling tokens:
        # Landmarks: Vibrant Pink/Coral (Hex: #EC4899 -> RGB: (236, 72, 153))
        # Connection lines: Electric Purple (Hex: #8B5CF6 -> RGB: (139, 92, 246))
        color_line = (139, 92, 246)  # RGB
        color_point = (236, 72, 153) # RGB
        
        for pose_landmarks in results.pose_landmarks:
            # 1. Draw connections (lines)
            for start_idx, end_idx in self.POSE_CONNECTIONS:
                if start_idx < len(pose_landmarks) and end_idx < len(pose_landmarks):
                    start_lm = pose_landmarks[start_idx]
                    end_lm = pose_landmarks[end_idx]
                    
                    # Verify both points are within visibility confidence threshold
                    if start_lm.visibility >= visibility_threshold and end_lm.visibility >= visibility_threshold:
                        # Convert normalized coordinates (0.0 to 1.0) to frame pixel values
                        start_point = (int(start_lm.x * w), int(start_lm.y * h))
                        end_point = (int(end_lm.x * w), int(end_lm.y * h))
                        cv2.line(frame_rgb, start_point, end_point, color_line, 2)
            
            # 2. Draw joints (landmark points)
            for lm in pose_landmarks:
                # Verify point is within visibility confidence threshold
                if lm.visibility >= visibility_threshold:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    # Keep circle drawings strictly within boundaries
                    if 0 <= cx < w and 0 <= cy < h:
                        # Draw a black border first for high contrast, then overlay color dot
                        cv2.circle(frame_rgb, (cx, cy), 5, (0, 0, 0), -1)
                        cv2.circle(frame_rgb, (cx, cy), 3, color_point, -1)
            
            # 3. Calculate and display joint angles
            # Target joints: Left/Right Elbow, Knee, and Hip
            # Structure: (Point A index, Point B vertex index, Point C index, Joint Name)
            angle_targets = [
                (11, 13, 15, "Left Elbow"),
                (12, 14, 16, "Right Elbow"),
                (23, 25, 27, "Left Knee"),
                (24, 26, 28, "Right Knee"),
                (11, 23, 25, "Left Hip"),
                (12, 24, 26, "Right Hip")
            ]
            
            for a_idx, b_idx, c_idx, name in angle_targets:
                if (a_idx < len(pose_landmarks) and 
                    b_idx < len(pose_landmarks) and 
                    c_idx < len(pose_landmarks)):
                    
                    a_lm = pose_landmarks[a_idx]
                    b_lm = pose_landmarks[b_idx]
                    c_lm = pose_landmarks[c_idx]
                    
                    # Only calculate/display angle if all three joints are visible
                    if (a_lm.visibility >= visibility_threshold and 
                        b_lm.visibility >= visibility_threshold and 
                        c_lm.visibility >= visibility_threshold):
                        
                        # Convert normalized coordinates to pixel coordinates
                        coord_a = (a_lm.x * w, a_lm.y * h)
                        coord_b = (b_lm.x * w, b_lm.y * h)
                        coord_c = (c_lm.x * w, c_lm.y * h)
                        
                        # Calculate angle using NumPy module
                        angle = calculate_angle(coord_a, coord_b, coord_c)
                        
                        bx, by = int(coord_b[0]), int(coord_b[1])
                        
                        # Text display setup
                        text_pos = (bx + 15, by - 5)
                        text_str = f"{int(angle)}°"
                        
                        # Draw outline (black) for high contrast readability
                        cv2.putText(
                            frame_rgb, 
                            text_str, 
                            text_pos, 
                            cv2.FONT_HERSHEY_DUPLEX, 
                            0.55, 
                            (0, 0, 0), 
                            3, 
                            cv2.LINE_AA
                        )
                        # Draw foreground text (white)
                        cv2.putText(
                            frame_rgb, 
                            text_str, 
                            text_pos, 
                            cv2.FONT_HERSHEY_DUPLEX, 
                            0.55, 
                            (255, 255, 255), 
                            1, 
                            cv2.LINE_AA
                        )
                
        return frame_rgb
        
    def get_landmarks_list(self, results, img_w, img_h):
        """
        Extract landmark positions as pixel coordinates list.
        Useful for downstream applications like rep counting or position checks.
        
        Args:
            results (PoseLandmarkerResult): Pose detection results.
            img_w (int): Width of the frame.
            img_h (int): Height of the frame.
            
        Returns:
            list: List of dict containing ID, x, y, z, and visibility.
        """
        landmarks = []
        if results.pose_landmarks:
            # We fetch from first detected pose
            pose_landmarks = results.pose_landmarks[0]
            for idx, lm in enumerate(pose_landmarks):
                cx, cy = int(lm.x * img_w), int(lm.y * img_h)
                landmarks.append({
                    "id": idx,
                    "x": cx,
                    "y": cy,
                    "z": lm.z,
                    "visibility": lm.visibility
                })
        return landmarks
