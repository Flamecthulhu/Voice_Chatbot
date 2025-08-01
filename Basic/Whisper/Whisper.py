import whisper
import time

location = r"H:\Codes\Cortana\Basic\Whisper\output.wav"
start_time = time.time()
whisper_model_list = ["tiny", "base", "small", "medium", "turbo", "large"]
model = whisper.load_model("medium")
result = model.transcribe(location)
print(result["text"])
end_time = time.time()
print(f"Elapsed time: {end_time - start_time:.2f} seconds")