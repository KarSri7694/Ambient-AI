import pyaudio
from datetime import datetime
import wave
import uuid
import time
import os
import silero_vad
from silero_vad import VADIterator
import numpy as np
import torch

model = silero_vad.load_silero_vad()
vad = VADIterator(
    model=model,
    threshold=0.7,
    min_silence_duration_ms=2000 #to account for dialogue breaks and speaker change
    )

agent = pyaudio.PyAudio()
device_info = agent.get_default_input_device_info()
DEFAULT_SAMPLE_RATE = int(device_info.get('defaultSampleRate'))
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK = 512
FORMAT = pyaudio.paInt16
RECORD_SECONDS = 30
UPLOADS_DIR = 'uploads/'
frames = []
confidence_list = []
PRE_ROLL_MS = 400
NUM_PREVIOUS_SAMPLES = int(SAMPLE_RATE * PRE_ROLL_MS / 1000)
previous_frame = None
voice_started = False


def int2float(sound):
    max = np.abs(sound).max()
    sound = sound.astype('float32')
    if max>0:
        sound *= 1/32768
    sound = sound.squeeze()
    return sound

def callback(in_data: bytes, frame_count, time_info, status_flags):
    # Only append data. Keep this function extremely fast.
    # print(type(in_data))
    global previous_frame, voice_started
    audio_int16 = np.frombuffer(in_data, np.int16)
    audio_float32 = int2float(sound=audio_int16)
    confidence = vad.__call__(torch.from_numpy(audio_float32))
    if confidence is not None:
        print(confidence)
    confidence_list.append(confidence)
    if voice_started:
        frames.append(in_data)
    try:
        if confidence is not None:
            if confidence.get('start'):
                voice_started = True
                # Add pre-roll audio before start was detected by VAD.
                starting_frame_in_chunk = confidence.get('start')%CHUNK
                if starting_frame_in_chunk < NUM_PREVIOUS_SAMPLES:
                    chunks_from_previous_frame = NUM_PREVIOUS_SAMPLES - starting_frame_in_chunk
                    if previous_frame is not None:
                        frames.append(previous_frame[chunks_from_previous_frame*2:len(previous_frame)])
                    else:
                        frames.append(in_data[:starting_frame_in_chunk*2])
                    frames.append(in_data[:starting_frame_in_chunk*2])
                else:
                    frames.append(in_data[(starting_frame_in_chunk-NUM_PREVIOUS_SAMPLES)*2:starting_frame_in_chunk*2])
                #add live audio
                #continue adding until 'end' is not encountered
                return (None, pyaudio.paContinue)
            
            if confidence.get('end'):
                voice_started = False
                return (None,pyaudio.paComplete)
    except KeyError:
        pass
    previous_frame= in_data
    return (None, pyaudio.paContinue)

# Open stream


while True:
    try:
        FILENAME = os.path.join(UPLOADS_DIR, f"audio_record_{datetime.now().strftime('%d%m%Y_%H%M%S')}_{uuid.uuid4()}.wav" )
        stream = agent.open(
            rate=SAMPLE_RATE, 
            channels=CHANNELS, 
            format=FORMAT, 
            input=True,
            frames_per_buffer=CHUNK, 
            input_device_index=1,
            stream_callback=callback
        )
        print("* Recording started...")
        stream.start_stream()
        # Use the main thread to handle the timing
        while stream.is_active():
            time.sleep(1)
        if stream.is_active():
            stream.stop_stream()
        stream.close()
        print("* Recording finished. Cleaning up...")

        with wave.open(FILENAME, "wb") as f:
            f.setnchannels(CHANNELS)
            f.setsampwidth(agent.get_sample_size(FORMAT))
            f.setframerate(SAMPLE_RATE)
            f.writeframes(b''.join(frames))

        print(f"Saved to {FILENAME}")
        frames = []
        vad.reset_states()
    except KeyboardInterrupt:
        print("Interrupt received, terminating")
        stream.close()
        agent.terminate()
        break