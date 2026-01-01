import streamlit as st
import tempfile
import os
import cv2
from processor import process_video

st.set_page_config(page_title="Tennis Rally Extractor", layout="wide")

st.title("🎾 Tennis Rally Extractor")
st.markdown("Upload your tennis video to automatically extract rallies and remove downtime.")

# Sidebar for settings
with st.sidebar:
    st.header("Settings")
    confidence = st.slider("Player Detection Confidence", 0.1, 1.0, 0.5)
    min_rally_duration = st.number_input("Min Rally Duration (sec)", value=3.0)
    padding = st.number_input("Clip Padding (sec)", value=2.0)
    
    st.divider()
    st.info("Ensure you have a GPU enabled for faster processing.")

# Input method selection
input_method = st.radio("Select Input Method", ["Upload Video", "Local File Path"])

video_path = None

if input_method == "Upload Video":
    uploaded_file = st.file_uploader("Choose a video file", type=['mp4', 'mov', 'avi'])
    if uploaded_file is not None:
        # Save uploaded file to a temporary file
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        video_path = tfile.name

else:
    local_path = st.text_input("Enter absolute path to video file (e.g., C:/Videos/match.mp4)")
    if local_path and os.path.exists(local_path):
        video_path = local_path
    elif local_path:
        st.error("File not found. Please check the path.")

if video_path is not None:
    st.video(video_path)
    
    if st.button("Analyze & Extract Rallies", type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        with st.spinner("Processing video..."):
            try:
                # Create output directory
                output_dir = "output_clips"
                os.makedirs(output_dir, exist_ok=True)
                
                # Callback for progress
                def update_progress(p):
                    progress_bar.progress(p)
                    status_text.text(f"Processing: {int(p*100)}%")

                # Call the processor
                stats, clips = process_video(
                    video_path, 
                    output_dir, 
                    conf_threshold=confidence, 
                    min_duration=min_rally_duration,
                    padding=padding,
                    progress_callback=update_progress
                )
                
                st.success(f"Processing Complete! Found {len(clips)} rallies.")
                
                # Display Stats (Debug Info)
                with st.expander("Debug Statistics", expanded=True):
                    st.json(stats)
                    if stats.get("frames_with_people", 0) == 0:
                        st.warning("No people were detected. Try lowering the confidence threshold.")
                    elif stats.get("total_rallies", 0) == 0:
                        st.warning("People were detected, but no segments met the minimum duration. Try lowering the 'Min Rally Duration'.")
                
                # Display results
                st.subheader("Extracted Rallies")
                cols = st.columns(2)
                for i, clip_path in enumerate(clips):
                    with cols[i % 2]:
                        st.markdown(f"**Rally {i+1}**")
                        st.video(clip_path)
                        
            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.exception(e)
            finally:
                # Cleanup temp file
                try:
                    os.unlink(video_path)
                except:
                    pass
