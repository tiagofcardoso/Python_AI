from playsound import playsound
from google.cloud import texttospeech
import requests  # Or another HTTP library
import os
from google.cloud import speech_v1p1beta1 as speech

# Set your Google Cloud credentials (replace with your project details)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "path/to/your/credentials.json"


def transcribe_audio(audio_file):
    client = speech.SpeechClient()

    with open(audio_file, "rb") as f:
        content = f.read()

    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="en-US",  # Change to your desired language
    )

    response = client.recognize(config=config, audio=audio)
    text = ""
    for result in response.results:
        text += result.alternatives[0].transcript
    return text


# Example usage:
# Replace with your audio file.  Ensure it's a WAV file, 16kHz mono
audio_file = "audio.wav"
transcript = transcribe_audio(audio_file)
print(f"Transcript: {transcript}")


def get_ai_response(user_input):
    # Replace with your actual API call
    api_url = "YOUR_API_ENDPOINT"
    # Or other authentication method
    headers = {"Authorization": "AIzaSyCUuakVTFI3gqmn2HyQumZL2UO3Z75qAEI"}
    data = {"text": user_input}  # Adjust data structure as needed

    response = requests.post(api_url, headers=headers, json=data)
    response.raise_for_status()  # Raise an exception for bad status codes

    return response.json()["response"]  # Extract the response from the JSON


# Example usage
ai_answer = get_ai_response(transcript)
print(f"AI response: {ai_answer}")


def synthesize_speech(text, output_file):
    client = texttospeech.TextToSpeechClient()
    synthesis_input = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US", ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
    )  # Customize voice as needed
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.LINEAR16
    )

    response = client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )

    with open(output_file, "wb") as out:
        out.write(response.audio_content)


# Example Usage
output_audio = "response.wav"
synthesize_speech(ai_answer, output_audio)

# Play the synthesized speech
playsound(output_audio)

# ... (Import statements and API setup from above) ...


def conversational_ai(audio_file):
    transcript = transcribe_audio(audio_file)
    ai_answer = get_ai_response(transcript)
    output_audio = "response.wav"
    synthesize_speech(ai_answer, output_audio)
    playsound(output_audio)


# Example usage
conversational_ai("audio.wav")

