# whisper for lixemburgish model 
# for the moment, it only works with small model, due to memory issues on the raspberry pi 5 - 8gb memory

# Load model directly
import re
import whisper
import torch
import time

myAudioPath = "/home/marc/development/python/stt/audio/"
#MODEL_PATH = "/home/marc/development/python/stt/models/"
#MODEL_PATH = "/home/marc/development/python/stt/models/ZLSCompLing/whisper_large_lb_ZLS_v4_38h/pytorch_model.bin"
MODEL_PATH = "/home/marc/development/python/stt/models/steja/whisper-small-luxembourgish/pytorch_model.bin"

def hf_to_whisper_states(text): return (text
    .replace("model.", "")
    .replace("layers", "blocks")
    .replace("fc1", "mlp.0")
    .replace("fc2", "mlp.2")
    .replace("final_layer_norm", "mlp_ln")
    .replace(".self_attn.q_proj", ".attn.query")
    .replace(".self_attn.k_proj", ".attn.key")
    .replace(".self_attn.v_proj", ".attn.value")
    .replace(".self_attn_layer_norm", ".attn_ln")
    .replace(".self_attn.out_proj", ".attn.out")
    .replace(".encoder_attn.q_proj", ".cross_attn.query")
    .replace(".encoder_attn.k_proj", ".cross_attn.key")
    .replace(".encoder_attn.v_proj", ".cross_attn.value")
    .replace(".encoder_attn_layer_norm", ".cross_attn_ln")
    .replace(".encoder_attn.out_proj", ".cross_attn.out")
    .replace("decoder.layer_norm.", "decoder.ln.")
    .replace("encoder.layer_norm.", "encoder.ln_post.")
    .replace("embed_tokens", "token_embedding")
    .replace("encoder.embed_positions.weight", "encoder.positional_embedding")
    .replace("decoder.embed_positions.weight", "decoder.positional_embedding")
    .replace("layer_norm", "ln_post")
)

# Load HF Model
print("loading Model ",MODEL_PATH )
hf_state_dict = torch.load(MODEL_PATH, map_location=torch.device('cpu'))    # pytorch_model.bin file
print("Model loaded")

# Rename layers
print("renaming layers")
for key in list(hf_state_dict.keys())[:]:
    
    new_key = hf_to_whisper_states(key)
    hf_state_dict[new_key] = hf_state_dict.pop(key)

print(" initialising Model")
# Init Whisper Model and replace model weights
whisper_model = whisper.load_model('small')  # need to use the same size than the model used
print("model created")
whisper_model.load_state_dict(hf_state_dict)

print("transcribing")
# set timer
start = time.time()
result = whisper_model.transcribe("/home/marc/development/python/stt/audio/audio-test.mp3")
end = time.time()
print(f' The text in audio: \n {result["text"]}')
print(f' it took \n {end - start}')
