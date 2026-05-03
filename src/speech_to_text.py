import os
import shutil
import whisper
import imageio_ffmpeg

# Set up local ffmpeg for Whisper
ffmpeg_exe_path = imageio_ffmpeg.get_ffmpeg_exe()
bin_dir = os.path.join(os.path.dirname(__file__), "bin")
os.makedirs(bin_dir, exist_ok=True)
local_ffmpeg = os.path.join(bin_dir, "ffmpeg.exe")
if not os.path.exists(local_ffmpeg):
    shutil.copy2(ffmpeg_exe_path, local_ffmpeg)
os.environ["PATH"] += os.pathsep + bin_dir



def transcribe_audio(audio_path: str, model_size: str = "tiny") -> list:
    """
    Transcribes audio using standard OpenAI Whisper.
    
    Args:
        audio_path (str): Path to the extracted audio file.
        model_size (str): Size of the Whisper model to use ('tiny', 'base', 'small', 'medium', 'large').
        
    Returns:
        list: A list of dictionaries containing 'start', 'end', and 'text' for each segment.
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
    print(f"Loading Whisper model '{model_size}'...")
    model = whisper.load_model(model_size)
    
    print(f"Transcribing audio {audio_path}...")
    result = model.transcribe(audio_path)
    
    print(f"Detected language: {result.get('language', 'unknown')}")
    
    transcribed_segments = []
    for segment in result["segments"]:
        print(f"[{segment['start']:.2f}s -> {segment['end']:.2f}s] {segment['text']}")
        transcribed_segments.append({
            "start": segment['start'],
            "end": segment['end'],
            "text": segment['text'].strip()
        })
        
    return transcribed_segments

# Example Usage
if __name__ == "__main__":
    input_audio = "extracted_audio.wav"
    try:
        results = transcribe_audio(input_audio, model_size="tiny")
    except Exception as e:
        print(f"An error occurred: {e}")
