import whisper
#model = whisper.load_model("base")
model = whisper.load_model("steja/whisper-small-luxembourgish")
#model = whisper.load_model("sZLSCompLing/whisper_large_lb_ZLS_v4_38h")
result = model.transcribe("/home/marc/development/python/stt/audio/audio-test.mp3")
print(f' The text in video: \n {result["text"]}')
