import os
from moviepy.editor import AudioFileClip, ColorClip

# Load the synthesized speech
audio = AudioFileClip("speech.wav")

# Create a blank black video clip of the exact same duration
video = ColorClip(size=(640, 480), color=(0, 0, 0), duration=audio.duration)

# Attach the audio to the video
video = video.set_audio(audio)

# Export as MP4
video.write_videofile("sample_video.mp4", fps=24, logger=None)

# Cleanup
audio.close()
video.close()
os.remove("speech.wav")
print("Successfully generated sample_video.mp4!")
