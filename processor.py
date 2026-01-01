import cv2
import os
from ultralytics import YOLO
import torch
import numpy as np
from moviepy.video.io.VideoFileClip import VideoFileClip

def process_video(video_path, output_dir, conf_threshold=0.5, min_duration=3.0, padding=2.0, progress_callback=None):
    """
    Process the video to extract rallies.
    
    Args:
        video_path (str): Path to input video.
        output_dir (str): Directory to save clips.
        conf_threshold (float): YOLO confidence threshold.
        min_duration (float): Minimum duration for a rally.
        padding (float): Seconds to add before/after rally.
        progress_callback (func): Function to update progress bar (0.0 to 1.0).
        
    Returns:
        tuple: (stats_dict, list_of_clip_paths)
    """
    # Load Model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    model = YOLO('yolov8n.pt') 
    model.to(device)

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
    out_of_frame_tolerance_frames = int(fps * 2.0)
    
    ball_detections_count = 0
    
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % skip_frames == 0:
            # We use DETECION_IMGSZ to help find the small ball
            results = model(frame, classes=TARGET_CLASSES, imgsz=DETECTION_IMGSZ, verbose=False)
            boxes = results[0].boxes
            
            people = [b for b in boxes if int(b.cls[0]) == 0 and b.conf[0] >= conf_threshold]
            balls = [b for b in boxes if int(b.cls[0]) == BALL_CLASS_ID and b.conf[0] >= BALL_CONF_THRESHOLD]
            
            has_ball = len(balls) > 0
            if has_ball:
                ball_detections_count += 1
                last_ball_seen_frame = frame_idx
                
                # Rally start logic: Ball detected and at least 1 person visible
                if not is_in_rally and len(people) >= 1:
                    is_in_rally = True
                    print(f"Rally started at frame {frame_idx} ({frame_idx/fps:.2f}s)")

            if is_in_rally:
                # Rally persists as long as the ball was seen recently (within 2s)
                if frame_idx - last_ball_seen_frame > out_of_frame_tolerance_frames:
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
    clip_paths = []
    
    if segments:
        try:
            with VideoFileClip(video_path) as video:
                for start_f, end_f in segments:
                    seg_duration = (end_f - start_f) / fps
                    if seg_duration >= min_duration:
                        start_time = max(0, (start_f / fps) - padding)
                        end_time = min(duration, (end_f / fps) + padding)
                        
                        final_clips.append((start_time, end_time))
                        clip_idx = len(final_clips)
                        output_filename = os.path.join(output_dir, f"rally_{clip_idx}.mp4")
                        
                        print(f"Extracting rally {clip_idx}: {start_time:.2f}s to {end_time:.2f}s")
                        
                        new = video.subclipped(start_time, end_time)
                        new.write_videofile(
                            output_filename, 
                            codec="libx264", 
                            audio_codec="aac", 
                            temp_audiofile=f'temp-audio-{clip_idx}.m4a', 
                            remove_temp=True
                        )
                        clip_paths.append(output_filename)
        except Exception as e:
            print(f"Error during extraction: {e}")

    stats["total_rallies"] = len(final_clips)
    return stats, clip_paths
