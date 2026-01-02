import cv2
import os
from ultralytics import YOLO
import torch
import numpy as np
from moviepy.video.io.VideoFileClip import VideoFileClip

def process_video(video_path, output_dir, original_filename=None, conf_threshold=0.5, min_duration=3.0, padding=2.0, ball_timeout=2.0, progress_callback=None):
    """
    Process the video to extract rallies.
    
    Args:
        video_path (str): Path to input video (might be a temp file).
        output_dir (str): Directory to save clips.
        original_filename (str, optional): The original name of the file if video_path is a temp file.
        conf_threshold (float): YOLO confidence threshold.
        min_duration (float): Minimum duration for a rally.
        padding (float): Seconds to add before/after rally.
        ball_timeout (float): Seconds to wait for ball to reappear before ending rally.
        progress_callback (func): Function to update progress bar (0.0 to 1.0).
        
    Returns:
        tuple: (stats_dict, list_of_clip_dicts)
    """
    # Load Models
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Object detection model for ball and person detection
    model = YOLO('yolov8n.pt') 
    model.to(device)
    
    # Pose estimation model for swing detection
    pose_model = YOLO('yolov8n-pose.pt')
    pose_model.to(device)
    print("Loaded object detection and pose estimation models")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    print(f"Video Info: {total_frames} frames, {fps} FPS, {duration:.2f}s")
    
    # YOLO Classes: 0 for person, 32 for tennis ball
    TARGET_CLASSES = [0, 32]
    
    # Optimizations for tennis ball (small object)
    BALL_CLASS_ID = 32
    BALL_CONF_THRESHOLD = 0.15 # Much lower for small fast moving objects
    DETECTION_IMGSZ = 1024     # Higher resolution for small objects
    
    skip_frames = 1 # Run on every frame for best tracking if GPU allows
    active_rally_frames = []
    
    # State tracking
    is_in_rally = False
    last_ball_seen_frame = -1
    last_swing_detected_frame = -1
    out_of_frame_tolerance_frames = int(fps * ball_timeout)  # Use configurable timeout
    
    ball_detections_count = 0
    swing_detections_count = 0
    
    # Pose keypoint indices (COCO format)
    # 5: left shoulder, 6: right shoulder
    # 7: left elbow, 8: right elbow
    # 9: left wrist, 10: right wrist
    
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % skip_frames == 0:
            # Object detection for ball and people
            results = model(frame, classes=TARGET_CLASSES, imgsz=DETECTION_IMGSZ, verbose=False)
            boxes = results[0].boxes
            
            people = [b for b in boxes if int(b.cls[0]) == 0 and b.conf[0] >= conf_threshold]
            balls = [b for b in boxes if int(b.cls[0]) == BALL_CLASS_ID and b.conf[0] >= BALL_CONF_THRESHOLD]
            
            has_ball = len(balls) > 0
            if has_ball:
                ball_detections_count += 1
                last_ball_seen_frame = frame_idx
            
            # Pose estimation for swing detection
            swing_detected = False
            if len(people) > 0:
                pose_results = pose_model(frame, verbose=False)
                if len(pose_results) > 0 and pose_results[0].keypoints is not None:
                    keypoints = pose_results[0].keypoints.data  # Shape: [num_people, 17, 3] (x, y, conf)
                    
                    for person_kp in keypoints:
                        # Check if we have valid arm keypoints
                        left_shoulder = person_kp[5]
                        right_shoulder = person_kp[6]
                        left_wrist = person_kp[9]
                        right_wrist = person_kp[10]
                        
                        # Check confidence of keypoints
                        if left_shoulder[2] > 0.5 and left_wrist[2] > 0.5:
                            # Check if left arm is raised (wrist above shoulder)
                            if left_wrist[1] < left_shoulder[1] - 30:  # Y decreases upward
                                swing_detected = True
                                break
                        
                        if right_shoulder[2] > 0.5 and right_wrist[2] > 0.5:
                            # Check if right arm is raised (wrist above shoulder)
                            if right_wrist[1] < right_shoulder[1] - 30:
                                swing_detected = True
                                break
            
            if swing_detected:
                swing_detections_count += 1
                last_swing_detected_frame = frame_idx
                
            # Rally start logic: Ball detected OR swing detected (with at least 1 person)
            if not is_in_rally and len(people) >= 1:
                if has_ball or swing_detected:
                    is_in_rally = True
                    trigger = "ball" if has_ball else "swing"
                    print(f"Rally started at frame {frame_idx} ({frame_idx/fps:.2f}s) - trigger: {trigger}")

            if is_in_rally:
                # Rally persists as long as ball was seen recently OR swing was recent
                frames_since_ball = frame_idx - last_ball_seen_frame
                frames_since_swing = frame_idx - last_swing_detected_frame
                
                if frames_since_ball > out_of_frame_tolerance_frames and frames_since_swing > out_of_frame_tolerance_frames:
                    is_in_rally = False
                    print(f"Rally ended at frame {frame_idx} ({frame_idx/fps:.2f}s) (timeout)")
                else:
                    for i in range(skip_frames):
                        if frame_idx + i < total_frames:
                            active_rally_frames.append(frame_idx + i)

        frame_idx += 1
        if frame_idx % 100 == 0:
            if progress_callback:
                progress_callback(frame_idx / total_frames)

    cap.release()
    if progress_callback:
        progress_callback(1.0)
    
    active_rally_frames = sorted(list(set(active_rally_frames)))
    
    stats = {
        "total_frames": total_frames,
        "frames_with_activity": len(active_rally_frames),
        "ball_detections": ball_detections_count,
        "swing_detections": swing_detections_count,
        "duration_processed": duration
    }
    
    if not active_rally_frames:
        return stats, []

    segments = []
    if active_rally_frames:
        start_f = active_rally_frames[0]
        prev_f = active_rally_frames[0]
        gap_limit = int(fps * 1.0) 
        
        for f in active_rally_frames[1:]:
            if f - prev_f > gap_limit:
                segments.append((start_f, prev_f))
                start_f = f
            prev_f = f
        segments.append((start_f, prev_f))
    
    stats["raw_segments_count"] = len(segments)
    
    final_clips = []
    
    if segments:
        # Get original filename without extension
        source_name = original_filename if original_filename else video_path
        original_basename = os.path.splitext(os.path.basename(source_name))[0]
        
        try:
            with VideoFileClip(video_path) as video:
                for start_f, end_f in segments:
                    seg_duration = (end_f - start_f) / fps
                    if seg_duration >= min_duration:
                        start_time = max(0, (start_f / fps) - padding)
                        end_time = min(duration, (end_f / fps) + padding)
                        actual_duration = end_time - start_time
                        
                        clip_idx = len(final_clips) + 1
                        
                        # New format: {original-file-name}-{start}-{duration}.mp4
                        filename = f"{original_basename}-{start_time:.1f}s-{actual_duration:.1f}s.mp4"
                        output_filename = os.path.join(output_dir, filename)
                        
                        print(f"Extracting rally {clip_idx}: {output_filename}")
                        
                        new = video.subclipped(start_time, end_time)
                        new.write_videofile(
                            output_filename, 
                            codec="libx264", 
                            audio_codec="aac", 
                            temp_audiofile=f'temp-audio-{clip_idx}.m4a', 
                            remove_temp=True
                        )
                        
                        final_clips.append({
                            "path": output_filename,
                            "start_time": start_time,
                            "duration": actual_duration,
                            "index": clip_idx
                        })
        except Exception as e:
            print(f"Error during extraction: {e}")

    stats["total_rallies"] = len(final_clips)
    return stats, final_clips
