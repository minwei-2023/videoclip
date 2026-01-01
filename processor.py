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
    
    skip_frames = 3
    active_frames = [] 
    
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % skip_frames == 0:
            results = model(frame, classes=[0], conf=conf_threshold, verbose=False)
            
            # Heuristic: At least 2 people (likely both players) should be detected for a rally
            if len(results[0].boxes) >= 2:
                # Mark this block of frames as active
                for i in range(skip_frames):
                    if frame_idx + i < total_frames:
                        active_frames.append(frame_idx + i)
        
        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"Processed {frame_idx}/{total_frames}")
            if progress_callback:
                progress_callback(frame_idx / total_frames)

    cap.release()
    if progress_callback:
        progress_callback(1.0)
    
    # Post-process
    active_frames = sorted(list(set(active_frames)))
    
    stats = {
        "total_frames": total_frames,
        "frames_with_people": len(active_frames),
        "duration_processed": duration
    }
    
    if not active_frames:
        print("No people detected in any frame.")
        return stats, []

    # Group into segments
    segments = []
    start_f = active_frames[0]
    prev_f = active_frames[0]
    
    # Increased gap tolerance to 2.0s to avoid splitting rallies
    gap_tolerance_frames = int(fps * 2.0) 
    
    for f in active_frames[1:]:
        if f - prev_f > gap_tolerance_frames:
            segments.append((start_f, prev_f))
            start_f = f
        prev_f = f
    segments.append((start_f, prev_f))
    
    stats["raw_segments_count"] = len(segments)
    
    # Filter by duration and add padding
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
                        
                        # MoviePy 2.x uses 'subclipped' instead of 'subclip'
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
