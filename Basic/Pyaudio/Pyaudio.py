import pyaudio
import wave

# Set parameters for recording
format = pyaudio.paInt16  # 16-bit resolution
channels = 1              # Mono audio
rate = 44100              # Sample rate (44.1 kHz)
chunk = 1024              # Size of each audio chunk
record_seconds = 5        # Record for 5 seconds
output_filename = "output.wav"

# Initialize the PyAudio object
p = pyaudio.PyAudio()

# Open the audio stream
stream = p.open(format=format,
                channels=channels,
                rate=rate,
                input=True,
                frames_per_buffer=chunk)

print("Recording...")

frames = []

# Record audio in chunks and append to frames
for i in range(0, int(rate / chunk * record_seconds)):
    data = stream.read(chunk)
    frames.append(data)

print("Recording finished.")

# Stop the audio stream
stream.stop_stream()
stream.close()
p.terminate()

# Save the recorded frames as a .wav file
with wave.open(output_filename, 'wb') as wf:
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(format))
    wf.setframerate(rate)
    wf.writeframes(b''.join(frames))

print(f"Audio saved as {output_filename}.")
