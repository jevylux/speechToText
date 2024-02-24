    - Objectif : test an stt ( speach to text) with the Lemswasabi/wav2vec2-large-xlsr-53-842h-luxembourgish-14h Model from Hugging Face

    - make a directory to contain all virtualenvironments
mkdir /home/marc/virtualenvironments
cd /home/marc/virtualenvironments
    - now create a virtual environment called stt
python3 -m venv stt
    - now the directory /home/marc/virtualenvironments/stt should be created
    - to select thos new virtual python environment use :
source /home/marc/virtualenvironments/stt/bin/activate  
    - (stt) should be displayed in front if the prompt
    - go back to our development directory cd /home/marc/development/python/stt/
    - install the dependencies : torch, torchaudio , transformers 
pip3 install torch, torchaudio, transformers, huggingface_hub
    - create some wav audio files and store them in the ./audio directtory
    - run stt-lux.py


    - models
steja/whisper-small-luxembourgish
Lemswasabi/wav2vec2-large-xlsr-53-842h-luxembourgish-4h
Lemswasabi/wav2vec2-large-xlsr-53-842h-luxembourgish-14h
Lemswasabi/wav2vec2-large-xlsr-53-842h-luxembourgish-11h
Lemswasabi/wav2vec2-large-xlsr-53-842h-luxembourgish-14h-with-lm
Lemswasabi/wav2vec2-base-librispeech-LS960h-LB842h-luxembourgish-4h

ZLSCompLing/whisper_large_lb_ZLS_v4_38h


    - translate
michaelfeil/ct2fast-m2m100_418M


    - models are cached here : /home/marc/.cache/huggingface/hub