# Voice Chatbot
-----

# Voice Assistant

This project is a simple voice assistant built using several powerful Python libraries and models for speech-to-text, large language model-based responses, and text-to-speech. The assistant wakes up when it hears a specific keyword, transcribes your speech, generates a text response, and then speaks the response back to you.

## Features

  * **Wake Word Detection:** Uses `pvporcupine` to listen for a specific keyword ("americano" by default) to activate the assistant.
  * **Speech-to-Text (STT):** Leverages OpenAI's `whisper` model to transcribe recorded audio into text.
  * **Large Language Model (LLM):** Employs a Hugging Face `transformers` model (`gpt2-xl` by default) to generate a conversational response based on the transcribed text.
  * **Text-to-Speech (TTS):** Utilizes `microsoft/speecht5_tts` to convert the LLM's text response back into spoken audio.
  * **Cross-Platform:** Works on systems with or without a CUDA-enabled GPU.

## Prerequisites

Before running the script, you'll need to install the required Python libraries.

```bash
pip install numpy pyaudio pvporcupine soundfile sounddevice torch whisper-openai transformers datasets scipy
```

You will also need to provide a valid Porcupine **Access Key** from the Picovoice Console. Replace the placeholder in the code with your key.

```python
access_key = "YOUR_ACCESS_KEY_HERE"
```

The script also assumes you have a directory named `audio files` containing `.wav` files. These files are used for the "wake-up" sound the assistant plays after detecting the keyword.

## Configuration

You can customize the voice assistant's behavior by modifying the following variables in the script:

  * `whisper_model_name`: The Whisper model size. Options include `"tiny"`, `"base"`, `"small"`, `"medium"`, `"turbo"`, and `"large"`. Smaller models are faster but less accurate.
  * `llm_model_name`: The LLM model to use. Options include `"gpt2"`, `"gpt2-medium"`, `"gpt2-large"`, `"gpt2-xl"`, and others available on Hugging Face.
  * `keywords`: The wake word(s) the assistant will listen for. The default is `["americano"]`. You can find a full list of available keywords in the Porcupine documentation.
  * `llm_length`: The maximum length of the generated response from the LLM.
  * `llm_temperature`: A value between 0 and 1 that controls the randomness of the LLM's output. Higher values lead to more creative but potentially nonsensical responses.

## How to Run

1.  **Install Prerequisites:** Make sure all the necessary libraries are installed.
2.  **Add Your Access Key:** Insert your Porcupine access key into the `access_key` variable.
3.  **Prepare Audio Files:** Create an `audio files` directory and place a few short `.wav` files inside.
4.  **Execute the Script:** Run the Python file from your terminal.

<!-- end list -->

```bash
python your_script_name.py
```

The console will print the configuration and then `Ready`. Once you say the keyword (`"americano"`), the assistant will play a sound, listen for your command, and then respond.

## How It Works

1.  **Initialization:** The script loads all the necessary models for Whisper, the LLM, and TTS. It also sets up `pyaudio` to listen to the microphone.
2.  **Wake Word Loop:** The program enters an infinite loop, continuously listening for the keyword using `pvporcupine`.
3.  **Activation:** When the keyword is detected, it plays a random audio file from the `audio files` directory.
4.  **Recording:** The script begins recording audio and continues until it detects a 2-second period of silence (indicating you've finished speaking).
5.  **Transcription:** The recorded audio is passed to the Whisper model, which converts it into a text string.
6.  **Response Generation:** The transcribed text is sent to the LLM (`gpt2-xl`), which generates a relevant text response.
7.  **Speech Synthesis:** The LLM's response is passed to the `speecht5_tts` model, which synthesizes it into a `.wav` file.
8.  **Playback:** The synthesized audio is played back through your speakers.
9.  **Repeat:** The program returns to the wake-word loop, waiting for the next command.