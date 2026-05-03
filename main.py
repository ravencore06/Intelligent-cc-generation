import os
import argparse
from src.audio_extractor import extract_audio
from src.speech_to_text import transcribe_audio
from src.subtitle_generator import generate_srt

def main():
    parser = argparse.ArgumentParser(description="Intelligent CC Generation MVP")
    parser.add_argument("video_path", type=str, help="Path to the input video file")
    parser.add_argument("--model", type=str, default="tiny", help="Whisper model size (tiny, base, small)")
    args = parser.parse_args()

    input_video = args.video_path
    
    if not os.path.exists(input_video):
        print(f"Error: The file {input_video} does not exist.")
        return

    # Set up paths
    base_name = os.path.splitext(os.path.basename(input_video))[0]
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    audio_path = os.path.join(output_dir, f"{base_name}.wav")
    srt_path = os.path.join(output_dir, f"{base_name}.srt")

    try:
        # Step 1: Video Input -> Audio Extraction
        extract_audio(input_video, audio_path)
        
        # Step 2: Audio -> Speech to Text
        segments = transcribe_audio(audio_path, model_size=args.model)
        
        # Step 3: Text -> Subtitle Generation
        generate_srt(segments, srt_path)
        
        print(f"\nSuccess! Subtitles have been saved to: {srt_path}")
        
    except Exception as e:
        print(f"An error occurred during processing: {e}")
        
    finally:
        # Optional: Clean up the intermediate audio file
        if os.path.exists(audio_path):
            os.remove(audio_path)
            print("Cleaned up temporary audio file.")

if __name__ == "__main__":
    main()
