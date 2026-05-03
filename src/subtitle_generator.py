def format_time(seconds: float) -> str:
    """
    Converts seconds into SRT time format (HH:MM:SS,ms).
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millisecs = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"

def generate_srt(segments: list, output_srt_path: str) -> str:
    """
    Generates an SRT (SubRip Subtitle) file from transcribed segments.
    
    Args:
        segments (list): List of dictionaries with 'start', 'end', and 'text'.
        output_srt_path (str): Path to save the SRT file.
        
    Returns:
        str: Path to the generated SRT file.
    """
    print(f"Generating SRT file at {output_srt_path}...")
    
    with open(output_srt_path, 'w', encoding='utf-8') as srt_file:
        for i, segment in enumerate(segments, start=1):
            start_time = format_time(segment["start"])
            end_time = format_time(segment["end"])
            text = segment["text"]
            
            srt_file.write(f"{i}\n")
            srt_file.write(f"{start_time} --> {end_time}\n")
            srt_file.write(f"{text}\n\n")
            
    print("Subtitle generation complete.")
    return output_srt_path
