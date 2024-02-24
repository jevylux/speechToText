# Load model directly
from transformers import AutoProcessor, AutoModelForCTC

processor = AutoProcessor.from_pretrained("Lemswasabi/wav2vec2-large-xlsr-53-842h-luxembourgish-14h-with-lm")
model = AutoModelForCTC.from_pretrained("Lemswasabi/wav2vec2-large-xlsr-53-842h-luxembourgish-14h-with-lm")