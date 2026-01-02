import streamlit as st
import tempfile
import os
import cv2
from processor import process_video

st.set_page_config(page_title="Tennis Rally Extractor", layout="wide")

st.title("🎾 Tennis Rally Extractor")
st.markdown("Upload your tennis video to automatically extract rallies and remove downtime.")

# Initialize session state for rallies
if "rallies" not in st.session_state:
    st.session_state.rallies = []
if "selected_rally" not in st.session_state:
    st.session_state.selected_rally = None
if "main_video_start" not in st.session_state:
    st.session_state.main_video_start = 0

# Sidebar for settings
with st.sidebar:
    st.header("Settings")
    confidence = st.slider("Player Detection Confidence", 0.1, 1.0, 0.5)
    min_rally_duration = st.number_input("Min Rally Duration (sec)", value=3.0)
    padding = st.number_input("Clip Padding (sec)", value=2.0)
    ball_timeout = st.number_input("Ball Out-of-Frame Timeout (sec)", value=4.0, min_value=0.5, max_value=5.0, step=0.5, 
                                    help="How long to wait for the ball to reappear before ending the rally")
    
    st.divider()
    if st.button("Clear Results"):
        st.session_state.rallies = []
        st.session_state.selected_rally = None
        st.session_state.main_video_start = 0
        st.rerun()

# Input method selection
input_method = st.radio("Select Input Method", ["Upload Video", "Local File Path"])

video_path = None
original_name = None

if input_method == "Upload Video":
    uploaded_file = st.file_uploader("Choose a video file", type=['mp4', 'mov', 'avi'])
    if uploaded_file is not None:
        original_name = uploaded_file.name
        # Use a more stable temp path if we want to sync
        temp_dir = "temp_uploads"
        os.makedirs(temp_dir, exist_ok=True)
        video_path = os.path.join(temp_dir, f"temp_{original_name}")
        if not os.path.exists(video_path):
            with open(video_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

else:
    local_path = st.text_input("Enter absolute path to video file (e.g., C:/Videos/match.mp4)")
    if local_path and os.path.exists(local_path):
        video_path = local_path
        original_name = os.path.basename(local_path)
    elif local_path:
        st.error("File not found. Please check the path.")

if video_path is not None:
    if st.button("Analyze & Extract Rallies", type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        with st.spinner("Processing video..."):
            try:
                output_dir = "output_clips"
                os.makedirs(output_dir, exist_ok=True)
                
                def update_progress(p):
                    progress_bar.progress(p)
                    status_text.text(f"Processing: {int(p*100)}%")

                stats, clips = process_video(
                    video_path, 
                    output_dir, 
                    original_filename=original_name,
                    conf_threshold=confidence, 
                    min_duration=min_rally_duration,
                    padding=padding,
                    ball_timeout=ball_timeout,
                    progress_callback=update_progress
                )
                
                st.session_state.rallies = clips
                st.session_state.stats = stats
                st.success(f"Found {len(clips)} rallies.")
                st.rerun() # Refresh to show results
                        
            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.exception(e)

# Rally Review Section
if st.session_state.rallies:
    st.divider()
    st.header("🎾 Rally Review")
    
    # 3-column layout: Original Video | Clip Player | Rally List
    col_original, col_clip, col_list = st.columns([1.5, 1.5, 1])
    
    with col_original:
        st.subheader("Original Video")
        st.video(video_path, start_time=st.session_state.main_video_start)
    
    with col_clip:
        st.subheader("Clip Player")
        if st.session_state.selected_rally:
            rally = st.session_state.selected_rally
            st.info(f"Rally #{rally['index']} - {rally['start_time']:.1f}s ({rally['duration']:.1f}s)")
            st.video(rally['path'])
        else:
            st.write("Select a rally from the list →")
    
    with col_list:
        st.subheader("Rallies")
        # Display clickable list
        for rally in st.session_state.rallies:
            label = f"#{rally['index']} ({rally['start_time']:.1f}s)"
            if st.button(label, key=f"btn_{rally['index']}", use_container_width=True):
                st.session_state.selected_rally = rally
                st.session_state.main_video_start = int(rally['start_time'])
                st.rerun()

    with st.expander("Debug Statistics"):
        if "stats" in st.session_state:
            st.json(st.session_state.stats)
