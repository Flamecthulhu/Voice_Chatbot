import pyaudio
import numpy as np
import pvporcupine
import wave

access_key = ""
porcupine = pvporcupine.create(keywords=["jarvis"], access_key=access_key)
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, 
                channels=1, 
                rate=porcupine.sample_rate, 
                input=True, 
                frames_per_buffer=porcupine.frame_length)

print("Listening for wake word...")

while True:
    audio = np.frombuffer(stream.read(porcupine.frame_length), dtype=np.int16)

    keyword_index = porcupine.process(audio)
    
    if keyword_index >= 0:
        print(f"Wake word detected")

stream.stop_stream()
stream.close()
p.terminate()
