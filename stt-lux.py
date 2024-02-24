import torchaudio
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# Load the pre-trained model and processor
processor = Wav2Vec2Processor.from_pretrained("Lemswasabi/wav2vec2-large-xlsr-53-842h-luxembourgish-14h")
model = Wav2Vec2ForCTC.from_pretrained("Lemswasabi/wav2vec2-large-xlsr-53-842h-luxembourgish-14h")

myAudioPath = "/home/marc/development/python/stt/audio/"
# Define a function for transcribing speech
def transcribe_audio(audio_path):
    # Load the audio file
    audio_input, _ = torchaudio.load(audio_path)
    # Preprocess the audio input
    input_values = processor(audio_input, return_tensors="pt").input_values
    # Transcribe the speech
    with torch.no_grad():
        logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)
    return transcription[0]

# Example usage
if __name__ == "__main__":
    audio_path = myAudioPath+"audio-test.wav"
    transcription = transcribe_audio(audio_path)
    print("Transcription:", transcription)