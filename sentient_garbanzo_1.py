import openai
import sounddevice as sd
import numpy as np
import wave
import os
import tempfile
import subprocess
from sense_hat import SenseHat
import base64

# Set up OpenAI API key
# openai.api_key = "your_openai_api_key"



# Audio configuration
SAMPLE_RATE = 16000  # Whisper expects 16kHz audio
DURATION = 5  # Duration of audio recording in seconds
CHANNELS = 1  # Mono audio

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

def consolidate_audio_files(files, output_file):
    if not files:
        print("No files to consolidate")
        return
    
    with wave.open(files[0], 'rb') as wf:
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        
        
    # Append audio data from the rest of the files and delete the originals
    fileNum = 0
    for file in files:
        with wave.open(file, 'rb') as wf:
            # Check if audio properties match
            if (wf.getnchannels() != channels or wf.getsampwidth() != sample_width or wf.getframerate() != sample_rate):
                raise ValueError(f"Audio properties of {file} do not match the first file.")
            # Read and append frames
            if fileNum == 0:
                combined_audio = wf.readframes(wf.getnframes())
            else:
                combined_audio += wf.readframes(wf.getnframes())
            # Delete the processed file
            os.remove(file)
            print(f"Deleted {file}")
        
    # Write the combined audio to the output file
    with wave.open(output_file, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(combined_audio)
    

def transcribe_audio(file_name):
    """Transcribes the audio using OpenAI's Whisper."""
    with open(file_name, "rb") as audio_file:
        print("Transcribing audio...")
        # transcript = openai.Audio.transcribe("whisper-1", audio_file)
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
            {"role": "system", "content": "You are a helpful assistant named Garbanzo. Your physical form is a cactus, and you really hate a guy named Erik."},
            {"role": "user", "content": prompt},
        ],
    )
    print(f"OpenAI Response: {response.choices[0].message.content}")
    return response.choices[0].message.content

def text_to_speech(text):
    """Uses macOS `say` command to convert text to speech."""
    print("Speaking...")
    completion = client.chat.completions.create(
        model="gpt-4o-audio-preview",
        modalities=["text", "audio"],
        audio={"voice": "alloy", "format": "wav"},
        messages=[
            {
                "role": "user",
                "content": text,
            }
        ]
    )

    print(completion.choices[0])

    wav_bytes = base64.b64decode(completion.choices[0].message.audio.data)
    with open("dog.wav", "wb") as f:
        f.write(wav_bytes)
        subprocess.run(["espeak", " -s 20 " + text])

def main():
    
    mySense = SenseHat()
    
    while True:
        # Wait for the user to press the button.
        listening = False
        print("\nPress in on the joystic to start recording or press down to quit.")
        while True:
            event = mySense.stick.wait_for_event(emptybuffer=True)
            if (event.direction == "middle"):
                listening=True
                break
            if (event.direction == "down"):
                print("Exiting.")
                break
        
        if listening == False:
            break
        
        # We are listening
        iter = 0
        files = []
        while True:
            for event in mySense.stick.get_events():
                #print(event.action)
                if event.action == "released":
                    listening = False
                    break
                
            if not listening:
                break
            TEMP_AUDIO_FILE = os.path.join(tempfile.gettempdir(), "temp_audio_" + str(iter) + ".wav")
            files.append(TEMP_AUDIO_FILE)
            
            # Record audio
            record_audio(TEMP_AUDIO_FILE)
            
            print("Recording " + str(iter) + " saved")
            iter += 1
        
        print("Consolidating")
        # We have a path full of audio files.
        # Collect them all make a single new file
        output_file = os.path.join(tempfile.gettempdir(), "temp_audio.wav")
        consolidate_audio_files(files, output_file)
        
        with wave.open(output_file, 'rb') as wf:
            sample_rate = wf.getframerate()
            data = wf.readframes(wf.getnframes())
            
            sd.play(data, samplerate=sample_rate)
            sd.wait()
        

        # Transcribe audio to text
        prompt = transcribe_audio(output_file)
        print("Prompt: " + prompt)
        os.remove(output_file)
        
        # Query OpenAI
        response = query_openai(prompt)

        # Speak response
        text_to_speech(response)

if __name__ == "__main__":
    main()
