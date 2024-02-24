from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import torch
import soundfile as sf
from datasets import load_dataset

processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
#processor = SpeechT5Processor.from_pretrained("mbarnig/lb-de-fr-en-pt-coqui-vits-tts")
#model = SpeechT5ForTextToSpeech.from_pretrained("mbarnig/lb-de-fr-en-pt-coqui-vits-tts")
#vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")microsoft/speecht5_tts

inputs = processor(text="hallo kolleeginen a kolleegen, aus dem Jevi senger Bastelbuud. Haut mat engem neie video deen iech hoffentlech e bësse freed mécht", return_tensors="pt")

# load xvector containing speaker's voice characteristics from a dataset

embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)

sf.write("./audio/out/speech.wav", speech.numpy(), samplerate=16000)