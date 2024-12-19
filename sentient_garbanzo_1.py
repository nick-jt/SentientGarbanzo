import openai
import sounddevice as sd
import numpy as np
import wave
import os
import tempfile
import subprocess
from sense_hat import SenseHat
import speech_recognition as sr
import noisereduce as nr
from pydub import AudioSegment
from pydub.silence import split_on_silence
import base64
import simpleaudio as sa
import io
import time

# Set up OpenAI API key
# openai.api_key = "your_openai_api_key"



# Audio configuration
SAMPLE_RATE = 16000  # Whisper expects 16kHz audio
DURATION = 3  # Duration of audio recording in seconds
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
        fileNum += 1
        
    # Write the combined audio to the output file
    with wave.open(output_file, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(combined_audio)
    
def preprocess_audio(input_file, output_file):    
    """Preprocess audio by removing silence and reducing noise."""
    with wave.open(input_file, 'rb') as wf:
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
    
    
    # Step 1: Remove Silence
    audio = AudioSegment.from_wav(input_file)
    chunks = split_on_silence(
        audio,
        min_silence_len=1000,  # Silence length in ms
        silence_thresh=-40     # Silence threshold in dBFS
    )
    
    cleaned_audio = AudioSegment.empty()
    for chunk in chunks:
        cleaned_audio += chunk

    # Save the intermediate cleaned file
    intermediate_file = "intermediate_cleaned.wav"
    cleaned_audio.export(intermediate_file, format="wav")

    # Step 2: Reduce Noise
    with wave.open(intermediate_file, 'rb') as wf:
        sample_rate = wf.getframerate()
        num_frames = wf.getnframes()
        audio_data = wf.readframes(num_frames)
        
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        reduced_audio = nr.reduce_noise(y=audio_array, sr=sample_rate)

    # Save the final preprocessed file
    with wave.open(output_file, 'wb') as wf_out:
        wf_out.setnchannels(channels)  # Mono audio
        wf_out.setsampwidth(sample_width)  # 16-bit audio
        wf_out.setframerate(sample_rate)
        wf_out.writeframes(reduced_audio.astype(np.int16).tobytes())

    print(f"Preprocessed audio saved to {output_file}")

def transcribe_audio(file_name):
    """Transcribes the audio using OpenAI's Whisper after preprocessing."""
    # Preprocess the audio
    #playAudio(file_name)
    
    #preprocessed_file = "preprocessed_audio.wav"
    #preprocess_audio(file_name, preprocessed_file)
    
    # playAudio(file_name)    

    # Transcribe the preprocessed audio
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
        model="gpt-4o-audio-preview",
        modalities=["text", "audio"],
        audio={"voice": "onyx", "format": "wav"},
        messages=[
            {"role": "system", "content": "You're a sassy little hoe named Garbanzo. Keep it quick, honey. Your physical form is a cactus that loves to dance, but not line dancing. You hate line dancing, and everybody that does line dance. You hate this guy Erik, and always talk about him because he likes to line dance."},
            {"role": "user", "content": prompt},
        ],
)
    print(f"OpenAI Response: {response.choices[0].message.content}")
    
    # We have a respnse, get the audio data and save the file
    filename = "dog.wav"
    wav_bytes = base64.b64decode(response.choices[0].message.audio.data)
    with open(filename, "wb") as f:
        f.write(wav_bytes)
    wave_read = wave.open(filename, 'rb')
    wave_obj = sa.WaveObject.from_wave_read(wave_read)  # Load audio data into WaveObject
    play_obj = wave_obj.play()  # Play the audio
    play_obj.wait_done()  # Wait for playback to finish
    
    return response.choices[0].message.content

'''
def text_to_speech(text):
    """Uses macOS `say` command to convert text to speech."""
    print("Speaking...")
    completion = client.chat.completions.create(
        model="gpt-4o-audio-preview",
        modalities=["text", "audio"],
        audio={"voice": "onyx", "format": "wav"},
        messages=[
            {
                "role": "user",
                "content": text
            }
        ]
    )
    filename = "dog.wav"
    wav_bytes = base64.b64decode(completion.choices[0].message.audio.data)
    with open(filename, "wb") as f:
        f.write(wav_bytes)
    wave_read = wave.open(filename, 'rb')
    wave_obj = sa.WaveObject.from_wave_read(wave_read)  # Load audio data into WaveObject
    play_obj = wave_obj.play()  # Play the audio
    play_obj.wait_done()  # Wait for playback to finish
'''

def text_to_speech_legacy(text):
    """Uses macOS `say` command to convert text to speech."""
    print("Speaking...")
    subprocess.run(["espeak", " -s 20 " + text])

def playAudio(file):
    with wave.open(file, 'rb') as wf:
        sample_rate = wf.getframerate()  # Get sample rate
        num_frames = wf.getnframes()     # Get the number of frames
        
        # Read the raw byte data
        raw_data = wf.readframes(num_frames)
        
        # Get the number of channels and sample width (e.g., 2 for 16-bit PCM)
        num_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        
        # Determine the format based on sample width
        if sample_width == 2:
            dtype = np.int16
        elif sample_width == 4:
            dtype = np.int32
        else:
            raise ValueError(f"Unsupported sample width: {sample_width}")

        # Convert the raw byte data to a NumPy array
        data = np.frombuffer(raw_data, dtype=dtype)
        
        # Reshape the data for stereo or multi-channel audio if necessary
        if num_channels > 1:
            data = data.reshape(-1, num_channels)
        
        # Play the audio data
        sd.play(data, samplerate=sample_rate)
        sd.wait()

def setRing(sense, color):
        sense.set_pixel(0, 0, color)
        sense.set_pixel(1, 0, color)
        sense.set_pixel(2, 0, color)
        sense.set_pixel(3, 0, color)
        sense.set_pixel(4, 0, color)
        sense.set_pixel(5, 0, color)
        sense.set_pixel(6, 0, color)
        sense.set_pixel(7, 0, color)
        sense.set_pixel(7, 1, color)
        sense.set_pixel(7, 2, color)
        sense.set_pixel(7, 3, color)
        sense.set_pixel(7, 4, color)
        sense.set_pixel(7, 5, color)
        sense.set_pixel(7, 6, color)
        sense.set_pixel(7, 7, color)
        sense.set_pixel(6, 7, color)
        sense.set_pixel(5, 7, color)
        sense.set_pixel(4, 7, color)
        sense.set_pixel(3, 7, color)
        sense.set_pixel(2, 7, color)
        sense.set_pixel(1, 7, color)
        sense.set_pixel(0, 7, color)
        sense.set_pixel(0, 6, color)
        sense.set_pixel(0, 5, color)
        sense.set_pixel(0, 4, color)
        sense.set_pixel(0, 3, color)
        sense.set_pixel(0, 2, color)
        sense.set_pixel(0, 1, color)


def connect_bluetooth_device(device_mac, mySense):
    """
    Attempts to connect to a Bluetooth device by MAC address.
    """
    CONNECTED = False
    
    while not CONNECTED:
        # Start bluetoothctl process
        try:
            #mySense.show_message("Connecting to Bluetooth...", scroll_speed=0.04)
            subprocess.run(['bluetoothctl', 'connect', device_mac], check=True)
            mySense.show_message("Connected to Bluetooth!", scroll_speed=0.02)
            CONNECTED = True
        except subprocess.CalledProcessError:
            mySense.show_message('Bluetooth Not Connected. Push Left to Try Again...', scroll_speed=0.02)
            
            event = mySense.stick.wait_for_event(emptybuffer=True)
            if event.direction == "left":
                mySense.show_message("Retrying...", scroll_speed=0.02)
                time.sleep(2)
            else:
                mySense.show_message("Exiting Bluetooth Setup...", scroll_speed=0.02)
                break
    
    return CONNECTED

def main():
    
    mySense = SenseHat()
    
    red = (255, 0, 0)
    green = (0, 255, 0)
    blue = (0, 0, 255)
    
    # Wait 5 seconds before trying the wifi
    count = 0
    while count < 2:
        mySense.show_message("...")
        time.sleep(1)
        count = count+1


    WIFI = True
    CONNECTED = False
    while WIFI:
        ################ Check on WIFI ####################
        ps = subprocess.Popen(['iwgetid'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        try:
            output=subprocess.check_output(('grep', 'ESSID'), stdin=ps.stdout)
            outputSTR = str(output)
            mySense.show_message('Wifi: ' + outputSTR[19:len(outputSTR)-4], scroll_speed = 0.02)
            WIFI = False
            CONNECTED = True
            
        except subprocess.CalledProcessError:
            sense.show_message('Wifi Not Connected. Push Left to Try Again...', scroll_speed = 0.02)
                
            event = mySense.stick.wait_for_event(emptybuffer = True)
            if event.direction == "left":
                mySense.show_message("Retrying...")
                time.sleep(2)
                WIFI = True
            else:
                WIFI = False 
    
    if not CONNECTED:
        setRing(mySense, red)
        exit(0)
    
    
    # Now, do the same for bluetooth
    device_mac_address = "EC:73:79:02:C2:B6"
    if not connect_bluetooth_device(device_mac_address, mySense):
        # If connection fails, set a red ring and exit
        setRing(mySense, red)
        exit(0)
        
        
    INIT = False
    while True:
        if INIT:
            client = openai.OpenAI()
            INIT = False
        
        setRing(mySense, green)
        
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
                mySense.clear()
                INIT = True
                break
        
        if listening == False:
            continue
        
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
            TEMP_AUDIO_FILE = os.getcwd() + "/temp_audio_" + str(iter) + ".wav"
            files.append(TEMP_AUDIO_FILE)
            
            # Record audio
            record_audio(TEMP_AUDIO_FILE)
            
            print("Recording " + str(iter) + " saved")
            iter += 1
        
        print("Consolidating")
        
        setRing(mySense, blue)
        
        # We have a path full of audio files.
        # Collect them all make a single new file
        output_file = os.getcwd() + "/temp_audio.wav"
        consolidate_audio_files(files, output_file)
        
        

        # Transcribe audio to text
        prompt = transcribe_audio(output_file)
        print("Prompt: " + prompt)
        os.remove(output_file)
        
        # Query OpenAI
        response = query_openai(prompt)

        # Speak response
        # text_to_speech(response)
    
    exit(0)

if __name__ == "__main__":
    main()
