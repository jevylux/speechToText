# this programm uses the pipeline to transcribe an audiofile 
# additional dependencies : pyctcdecode, kenlm

from transformers import pipeline
import os
import time
import shutil

audio_file = "/home/marc/development/python/stt/audio/audio-long.wav"

transcriber = pipeline("automatic-speech-recognition", model="Lemswasabi/wav2vec2-large-xlsr-53-842h-luxembourgish-14h-with-lm")
print(transcriber(audio_file))

# This function will check incoming files in a directory
# if a new file is detected, we check that the file is not being modified
# when the file is stable, we will transcribe it's audio content 
# when handling is finished, we will move to file to a new directory

def load_new_file(directory, move_directory):
    # Get the list of files in the directory
    files = os.listdir(directory)
    
    # Check for new files
    for audio_file in files:
        time.sleep(1)  # Wait for 1 second
        current_modification_time = os.path.getmtime(os.path.join(directory, audio_file))
        time.sleep(1)  # Wait for another second
        new_modification_time = os.path.getmtime(os.path.join(directory, audio_file))
        if current_modification_time == new_modification_time:
            print(f"New file detected: {audio_file}")
            # Perform loading operations here
            print("transcribing")
            start = time.time()
            # for longer audio files, we will use chunks of data. otherwise the amount of memory used will be to high
            print(transcriber(directory+audio_file,chunk_length_s=10, stride_length_s=(4,2)))
            end = time.time()
            print(f' it took \n {end - start}')
            shutil.move(directory+audio_file, move_directory+audio_file)
        else:
            print(f"File {audio_file} is still being uploaded. Skipping...")

# run the application 

if __name__ == "__main__":
    # Directories to use
    directory_to_watch = "/home/marc/development/python/stt/audio/watch/"
    directory_to_move = "/home/marc/development/python/stt/audio/transcribed/"
    
    # handle incoming new audio files
    try:
        while True:
            load_new_file(directory_to_watch, directory_to_move)
            time.sleep(5)  # Adjust the delay as per your requirement
    except KeyboardInterrupt:
        print("Monitoring stopped.")
