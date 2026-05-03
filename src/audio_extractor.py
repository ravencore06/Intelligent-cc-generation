import os
from moviepy.editor import VideoFileClip

def extract_audio(video_path: str, output_audio_path: str) -> str:
    """
    Extracts the audio track from a video file and saves it as a WAV file.
    
    Args:
        video_path (str): Path to the input video file (e.g., .mp4, .avi).
        output_audio_path (str): Path to save the extracted audio file (e.g., .wav, .mp3).
        
    Returns:
        str: Path to the generated audio file.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
        
    print(f"Extracting audio from {video_path}...")
    
    # Load the video file
    video_clip = VideoFileClip(video_path)
    
    # Isolate the audio track
    audio_clip = video_clip.audio
    
    if audio_clip is None:
        raise ValueError(f"No audio track found in {video_path}")
    
    # Save the audio track to the output path
    # Note: 16000 Hz (16kHz) with mono channel is optimal for most ASR models like Whisper
    audio_clip.write_audiofile(
        output_audio_path, 
        fps=16000, 
        nbytes=2, 
        codec='pcm_s16le', # Use standard WAV encoding
        verbose=False, 
        logger=None # Set logger to 'bar' if you want to see a progress bar in terminal
    )
    
    # Important: Close the clips to free up system memory
    audio_clip.close()
    video_clip.close()
    
    print(f"Audio extracted successfully to {output_audio_path}")
    return output_audio_path

# Example Usage
if __name__ == "__main__":
    input_video = "sample_video.mp4"
    output_audio = "extracted_audio.wav"
    
    try:
        extract_audio(input_video, output_audio)
    except Exception as e:
        print(f"An error occurred: {e}")
