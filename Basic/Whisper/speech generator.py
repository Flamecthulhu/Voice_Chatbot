from transformers import pipeline, logging
from datasets import load_dataset
from num2words import num2words
import soundfile as sf
import torch
import time

logging.set_verbosity_error()

location = "Basic\Whisper\output.wav" #file name

synthesiser = pipeline("text-to-speech", "microsoft/speecht5_tts")

embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embedding = torch.tensor(embeddings_dataset[5]["xvector"]).unsqueeze(0)
'''
bdl (US male) → embeddings_dataset[0]["xvector"]
slt (US female) → embeddings_dataset[1]["xvector"]
jmk (Canadian male) → embeddings_dataset[2]["xvector"]
awb (Scottish male) → embeddings_dataset[3]["xvector"]
rms (US male) → embeddings_dataset[4]["xvector"]
clb (US female) → embeddings_dataset[5]["xvector"]
ksp (Indian male) → embeddings_dataset[6]["xvector"]
'''

if torch.cuda.is_available():
    synthesiser.model.to('cuda')
    speaker_embedding = speaker_embedding.to('cuda')
else:
    print("CUDA is not available. Running on CPU.")
    
while True:
    speech_text = input('User input: ')
    start_time = time.time()
    speech = synthesiser(speech_text, forward_params={"speaker_embeddings": speaker_embedding})
    sf.write(location, speech["audio"], samplerate=speech["sampling_rate"])
    print("Saved speech to output.wav")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")
    print()
    