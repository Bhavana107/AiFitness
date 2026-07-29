import yt_dlp
import os
import cv2

def download_youtube_video(url: str, output_dir: str = "videos/downloads", progress_callback=None) -> dict:
    """
    Downloads a YouTube video using yt-dlp.
    
    Args:
        url (str): YouTube URL.
        output_dir (str): Output directory for the video file.
        progress_callback (callable): Function to update download progress (takes float between 0.0 and 1.0).
        
    Returns:
        dict: Metadata containing 'file_path', 'title', 'thumbnail_url', 'duration'.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Progress hook for yt-dlp
    def hook(d):
        if d['status'] == 'downloading':
            total = d.get('total_bytes') or d.get('total_bytes_estimate')
            downloaded = d.get('downloaded_bytes', 0)
            if total and progress_callback:
                progress_callback(float(downloaded) / float(total))
                
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best', # Ensure MP4
        'outtmpl': os.path.join(output_dir, '%(id)s.%(ext)s'),
        'progress_hooks': [hook],
        'quiet': True,
        'no_warnings': True,
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        file_path = ydl.prepare_filename(info)
        
        # In case merging happened and changed the file extension, double check path
        if not os.path.exists(file_path):
            video_id = info.get('id')
            for f in os.listdir(output_dir):
                if f.startswith(video_id):
                    file_path = os.path.join(output_dir, f)
                    break
                    
        return {
            'file_path': file_path,
            'title': info.get('title', 'YouTube Video'),
            'thumbnail_url': info.get('thumbnail'),
            'duration': info.get('duration', 0)
        }

def analyze_youtube_video(video_path: str, output_path: str, detector, progress_callback=None) -> dict:
    """
    Processes a video frame-by-frame, runs MediaPipe Pose detector,
    renders skeletons, and saves the output video file.
    
    Args:
        video_path (str): Local path to downloaded input video.
        output_path (str): Output path for the processed video.
        detector (PoseDetector): The active PoseDetector instance.
        progress_callback (callable): Function to update detection progress (takes float).
        
    Returns:
        dict: Analysis results containing 'timeline_data' (list of dicts).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")
        
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames <= 0:
        total_frames = 1
        
    if fps <= 0:
        fps = 30.0 # Default fallback
        
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Standard mp4v
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    timeline_data = []
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            timestamp = frame_count / fps
            
            # Convert to RGB for MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect Pose landmarks
            results = detector.find_pose(frame_rgb)
            
            success = False
            if results.pose_landmarks:
                success = True
                # Draw skeleton landmarks on frame
                frame_rgb = detector.draw_skeleton(frame_rgb, results)
                
            timeline_data.append({
                "Timestamp (s)": round(timestamp, 2),
                "Tracking Status": 1.0 if success else 0.0
            })
            
            # Convert back to BGR for cv2 VideoWriter
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
            
            if progress_callback:
                progress_callback(frame_count / total_frames)
                
    finally:
        cap.release()
        out.release()
        
    return {
        "timeline_data": timeline_data,
        "total_frames": frame_count,
        "fps": fps,
        "duration": frame_count / fps
    }
