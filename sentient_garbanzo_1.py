import openai
import sounddevice as sd
import numpy as np
import wave
import os
import tempfile
import subprocess
import base64
import simpleaudio as sa
import io

# Set up OpenAI API key
# openai.api_key = "your_openai_api_key"

# Audio configuration
SAMPLE_RATE = 16000  # Whisper expects 16kHz audio
DURATION = 5  # Duration of audio recording in seconds
CHANNELS = 1  # Mono audio
TEMP_INPUT_AUDIO_FILE = os.path.join(tempfile.gettempdir(), "temp_audio_in.wav")
TEMP_OUTPUT_AUDIO_FILE = os.path.join(tempfile.gettempdir(), "temp_audio_out.wav")
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
        transcript = client.audio.transcriptions.create(
          model="whisper-1",
          file=audio_file
        )
    return transcript.text

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
    print(f"OpenAI Response: {response.choices[0].message.content}")
    return response.choices[0].message.content

def text_to_speech(text, filename):
    """Uses macOS `say` command to convert text to speech."""
    print("Speaking...")
    completion = client.chat.completions.create(
        model="gpt-4o-audio-preview",
        modalities=["text", "audio"],
        audio={"voice": "alloy", "format": "wav"},
        messages=[
            {
                "role": "user",
                "content": text
            }
        ]
    )
    wav_bytes = base64.b64decode(completion.choices[0].message.audio.data)
    with open(filename, "wb") as f:
        f.write(wav_bytes)
    wave_read = wave.open(filename, 'rb')
    wave_obj = sa.WaveObject.from_wave_read(wave_read)  # Load audio data into WaveObject
    play_obj = wave_obj.play()  # Play the audio
    play_obj.wait_done()  # Wait for playback to finish

def main():
    while True:
        print("\nPress Enter to start recording or type 'exit' to quit.")
        command = input("Command: ").strip().lower()
        if command == "exit":
            print("Exiting.")
            break

        # Record audio
        record_audio(TEMP_INPUT_AUDIO_FILE)

        # Transcribe audio to text
        prompt = transcribe_audio(TEMP_INPUT_AUDIO_FILE)

        # Query OpenAI
        response = query_openai(prompt)

        # Speak response
        text_to_speech(response, TEMP_OUTPUT_AUDIO_FILE)

if __name__ == "__main__":
    main()
