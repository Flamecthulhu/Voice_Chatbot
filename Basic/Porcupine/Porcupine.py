import pyaudio
import numpy as np
import pvporcupine
import wave

access_key = "aPeEK2Idipevi3QiZ0Lb+VMxNNzkfPYGjePmOqlXyT2MH3tPsLaKZg=="
# 初始化 Porcupine
porcupine = pvporcupine.create(keywords=["jarvis"], access_key=access_key)

# 初始化 PyAudio
p = pyaudio.PyAudio()

# 打開麥克風流
stream = p.open(format=pyaudio.paInt16, 
                channels=1, 
                rate=porcupine.sample_rate, 
                input=True, 
                frames_per_buffer=porcupine.frame_length)

print("Listening for wake word...")

while True:
    # 讀取音頻
    audio = np.frombuffer(stream.read(porcupine.frame_length), dtype=np.int16)

    # 檢查是否偵測到喚醒詞
    keyword_index = porcupine.process(audio)
    
    if keyword_index >= 0:
        print(f"Wake word detected")

# 關閉麥克風流
stream.stop_stream()
stream.close()
p.terminate()
