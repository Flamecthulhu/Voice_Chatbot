import numpy as np
import os
import pyaudio
import pvporcupine
import random
import soundfile as sf
import sounddevice as sd
import time
import torch
import wave
import whisper
from datasets import load_dataset
from scipy.io.wavfile import write
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, pipeline, logging

#Other config
loading_delay = 1

#File location config
record_location = "record_output.wav"
generated_location = "model_output.wav"
audio_file_directory = "audio files"

#Model config
whisper_model_list = ["tiny", "base", "small", "medium", "turbo", "large"]
whisper_model_name = "small"
llm_model_list = ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl", "Phi 3", "Phi 3.5"]
llm_model_name = "gpt2-xl"
llm_temperature = 1.0
llm_length = 30
tts_model_name = "microsoft/speecht5_tts"
keywords = ["americano"] #npicovoice, porcupine, blueberry, computer, hey siri, bumblebee, grasshopper, hey barista, jarvis, pico clock, hey google, grapefruit, terminator, alexa, ok google, americano

print("Detecting device configuration... ", end="", flush=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Done")
time.sleep(loading_delay)
logging.set_verbosity_error()
print("Loading Tokenizer... ", end="", flush=True)
llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
print("Done")
time.sleep(loading_delay)
print(f"Loading Model {llm_model_name}... ", end="", flush=True)
llm_model = AutoModelForCausalLM.from_pretrained(llm_model_name).to(device)
print("Done")
time.sleep(loading_delay)
llm_tokenizer.pad_token = llm_tokenizer.eos_token
generation_config = GenerationConfig(
    do_sample=True,
    num_beams=5,
    no_repeat_ngram_size=2,
    top_k=50,
    top_p=0.9
)
llm_model.eval()

access_key = ""

record_sample_rate = 16000

VAD_THRESHOLD = 1000
print(f"Loading {tts_model_name}... ", end="", flush=True)
synthesiser = pipeline("text-to-speech", tts_model_name)
print("Done")
time.sleep(loading_delay)
print("Loading dataset... ", end="", flush=True)
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embedding = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)
print("Done")
porcupine = pvporcupine.create(keywords=keywords, access_key=access_key)
whisper_model = whisper.load_model(whisper_model_name)
p = pyaudio.PyAudio()
stream = p.open(
    format=pyaudio.paInt16,
    channels=1,
    rate=porcupine.sample_rate,
    input=True,
    frames_per_buffer=porcupine.frame_length
)
print("Your configuration")
print(f"Compute unit: {device.upper()}\tWhisper model: {whisper_model_name}\tLanguage model: {llm_model_name.upper()}\tTTS model: {tts_model_name}\nMax Length: {llm_length}\t Temperature: {llm_temperature}\t Keyword: {keywords}")

def record_audio():
    recording = []
    start_time = time.time()
    while True:
        audio_data = np.frombuffer(stream.read(porcupine.frame_length), dtype=np.int16)
        audio_level = np.max(np.abs(audio_data))

        if audio_level > VAD_THRESHOLD:
            recording.append(audio_data)
            start_time = time.time()
            

        elif time.time() - start_time > 2:
            break
    
    recording = np.concatenate(recording, axis=0)
    recording = np.clip(recording, -32768, 32767).astype(np.int16)
    write(record_location, record_sample_rate, recording)

def play_audio(directory):
    audio_files = [f for f in os.listdir(directory) if f.endswith('.wav')]
    
    if not audio_files:
        return
    
    random_file = random.choice(audio_files)
    file_path = os.path.join(directory, random_file)
    with wave.open(file_path, 'rb') as wf:
        fs = wf.getframerate()
        frames = wf.readframes(wf.getnframes())
        audio_data = np.frombuffer(frames, dtype=np.int16)

    audio_data = np.int16(audio_data * 0.5)
    sd.play(audio_data, fs)
    sd.wait()

print("Ready")
while True:
    audio = np.frombuffer(stream.read(porcupine.frame_length), dtype=np.int16)
    keyword_index = porcupine.process(audio)
    if keyword_index >= 0:
        play_audio(audio_file_directory)
        record_audio()
        result = whisper_model.transcribe(record_location)
        print(result["text"])
        try:
            input_text = result["text"]
            inputs = llm_tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
            inputs = {key: value.to(device) for key, value in inputs.items()}
            output = llm_model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                do_sample=generation_config.do_sample,
                max_length=llm_length,
                temperature=llm_temperature,
                num_beams=generation_config.num_beams,
                no_repeat_ngram_size=generation_config.no_repeat_ngram_size,
                top_k=generation_config.top_k,
                top_p=generation_config.top_p,
                pad_token_id=llm_tokenizer.eos_token_id
            )
            responce = llm_tokenizer.decode(output[0], skip_special_tokens=True)
            print(responce.strip(input_text))
            print()
            speech_text = responce.strip(input_text)
            speech = synthesiser(speech_text, forward_params={"speaker_embeddings": speaker_embedding})
            speech_array = np.array(speech["audio"], dtype=np.float32)
            if len(speech_array.shape) == 1:
                speech_array = np.expand_dims(speech_array, axis=1)

            sf.write(generated_location, speech_array, samplerate=16000)
            data, samplerate = sf.read(generated_location)
            sd.play(data, samplerate)
            sd.wait()
        except Exception as e:
            print(f"Error: {e}")
            continue

stream.stop_stream()
stream.close()
p.terminate()