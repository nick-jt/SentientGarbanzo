import openai
import sounddevice as sd
import numpy as np
import wave
import os
import tempfile
import subprocess

# Set up OpenAI API key
# openai.api_key = "your_openai_api_key"

# Audio configuration
SAMPLE_RATE = 16000  # Whisper expects 16kHz audio
DURATION = 5  # Duration of audio recording in seconds
CHANNELS = 1  # Mono audio
TEMP_AUDIO_FILE = os.path.join(tempfile.gettempdir(), "temp_audio.wav")
client = openai.OpenAI()

def record_audio(file_name, duration=DURATION):
    """Records audio and saves it to a file."""
    print("Recording...")
    audio_data = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=CHANNELS, dtype=np.int16)
    sd.wait()  # Wait for the recording to finish
    print("Recording complete.")
    
    # Save audio to file
    with wave.open(file_name, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)  # 16-bit audio
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_data.tobytes())

def transcribe_audio(file_name):
    """Transcribes the audio using OpenAI's Whisper."""
    with open(file_name, "rb") as audio_file:
        print("Transcribing audio...")
        # transcript = openai.Audio.transcribe("whisper-1", audio_file)
        transcript = client.audio.transcriptions.create(
          model="whisper-1",
          file=audio_file
        )
    print(f"Transcription: {transcript}")
    return transcript

def query_openai(prompt):
    """Sends a prompt to OpenAI's GPT-4 and returns the response."""
    print("Querying OpenAI...")
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
    )
    answer = response['choices'][0]['message']['content']
    print(f"OpenAI Response: {answer}")
    return answer

def text_to_speech(text):
    """Uses macOS `say` command to convert text to speech."""
    print("Speaking...")
    subprocess.run(["say", text])

def main():
    while True:
        print("\nPress Enter to start recording or type 'exit' to quit.")
        command = input("Command: ").strip().lower()
        if command == "exit":
            print("Exiting.")
            break

        # Record audio
        record_audio(TEMP_AUDIO_FILE)

        # Transcribe audio to text
        prompt = transcribe_audio(TEMP_AUDIO_FILE)

        # Query OpenAI
        response = query_openai(prompt)

        # Speak response
        text_to_speech(response)

if __name__ == "__main__":
    main()
