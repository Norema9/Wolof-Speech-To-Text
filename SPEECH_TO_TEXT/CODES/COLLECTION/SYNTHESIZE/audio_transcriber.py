import os
import librosa
from pydub.silence import detect_nonsilent
import pandas as pd
import numpy as np
import torch
from transformers import Wav2Vec2ProcessorWithLM, Wav2Vec2ForCTC, Wav2Vec2Processor
from pydub import AudioSegment
import noisereduce as nr
import soundfile as sf 
from huggingface_hub import login
from tqdm import tqdm
import json

class Transcriber:
    _processor_lm_cache = None
    _processor_cache = None
    _model_cache = None

    def __init__(self, device, 
                silence_thresh = -50, 
                min_silence_len = 200,
                # selected_model="Model Wav2Vec2.0", 
                selected_model="Model Wav2Vec2.0 with LM",
                pretrained_model = "M9and2M/marone_wolof_wav2vec2-xls-r-300m_lm",
                # pretrained_model = r"/mnt/f/WOLOF/SPEECH_TO_TEXT/MODELS/WAV2VEC2/huggingface-hub_lm",
                # pretrained_model=r"F:\WOLOF\SPEECH_TO_TEXT\MODELS\WAV2VEC2\huggingface-hub_1", 
                key = "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"):
        """
        Initialize the Transcriber with the specified model and device.

        Parameters: 
        device (str): Device to run the model on ('cpu' or 'cuda').
        silence_threshold_db (float): Threshold for considering audio as silent.
        selected_model (str): Model type to use ('Model Wav2Vec2.0' or 'Model Wav2Vec2.0 with LM').
        pretrained_model (str): Pretrained model path or identifier.
        key (str): API key for Hugging Face Hub.
        """
        self.device = device
        login(token=key)
        
        if selected_model == "Model Wav2Vec2.0 with LM":
            if Transcriber._processor_lm_cache is None or Transcriber._model_cache is None:
                Transcriber._processor_lm_cache = Wav2Vec2ProcessorWithLM.from_pretrained(pretrained_model)
                Transcriber._model_cache = Wav2Vec2ForCTC.from_pretrained(pretrained_model).to(self.device)
            self.processor = Transcriber._processor_lm_cache
        else:
            if Transcriber._processor_cache is None or Transcriber._model_cache is None:
                Transcriber._processor_cache = Wav2Vec2Processor.from_pretrained(pretrained_model)
                Transcriber._model_cache = Wav2Vec2ForCTC.from_pretrained(pretrained_model).to(self.device)
            self.processor = Transcriber._processor_cache
            
        self.model = Transcriber._model_cache
        self.model.eval()
        self.selected_model = selected_model
        self.silence_thresh = silence_thresh
        self.min_silence_len = min_silence_len

    def transcribe(self, wav):
        """
        Transcribe the given audio waveform.

        Parameters:
        wav (numpy.ndarray): Audio waveform as a numpy array.

        Returns:
        str: Transcribed text.
        """
    
        input_values = self.processor(torch.tensor(wav), sampling_rate=16000, return_tensors="pt").input_values
        logits = self.model(input_values.to(self.device)).logits
    
        if self.selected_model == "Model Wav2Vec2.0 with LM":
            pred_transcript = self.processor.batch_decode(logits.detach().numpy()).text
        else:
            pred_ids = torch.argmax(logits, axis=-1)
            pred_transcript = self.processor.batch_decode(pred_ids)
            
        return pred_transcript[0]

    def split_audio(self, audio):
        """
        Split the audio into segments with no silent audio and durations between 0.1 and 6 seconds.

        Returns:
        list: A list of tuples where each tuple contains the start and end times of a non-silent segment.
        """
        segments = self.detect_and_filter_segments(audio, self.silence_thresh)

        return segments

    def detect_and_filter_segments(self, audio, silence_thresh):
        # Split audio on silence
        segments = detect_nonsilent(audio, min_silence_len=self.min_silence_len, silence_thresh=silence_thresh, seek_step=1)

        # Filter segments based on duration
        segments_filtered = [(start / 1000, end / 1000) for (start, end) in segments if 250 <= (end - start) <= 6000]
        long_segments = [(start / 1000, end / 1000) for (start, end) in segments if (end - start) > 6000]
        if long_segments:
            long_segment_filtered = self.process_long_segments(audio, long_segments, silence_thresh)
            segments_filtered.extend(long_segment_filtered)

        return segments_filtered

    def process_long_segments(self, audio, segments, silence_thresh):
        processed_segments = []
        for start, end in segments:
            # Re-split this segment with a higher silence threshold
            sub_segments = self.detect_and_filter_segments(audio[start*1000:end*1000], silence_thresh + 5)
            if sub_segments:
                # Adjust the time of sub-segments to the original audio
                sub_segments = [(start + sub_start, start + sub_end) for sub_start, sub_end in sub_segments]
                processed_segments.extend(sub_segments)
            else:
                # If no sub-segments were found, add the original segment
                processed_segments.append((start, end))
        return processed_segments
    
    def process_file(self, audios_path, file_path):
        """
        Process an audio file for transcription.

        Parameters:
        file_path (str): Path to the audio file.

        Returns:
        str: Transcribed text.
        """
        audio = AudioSegment.from_file(os.path.join(audios_path, file_path))
        segments = self.split_audio(audio)

        results = []
        save_path = os.path.join("audios", file_path)
        os.makedirs(save_path, exist_ok=True)

        for i, (start, end) in tqdm(enumerate(segments)):
            segment = audio[start * 1000:end * 1000]

            # Export the segment to a temporary file
            temp_audio_path = os.path.join(save_path, f"temp_segment_{i}.wav")
            segment.export(temp_audio_path, format="wav")
            
            transcript = self.transcribe(librosa.load(temp_audio_path, sr=16000)[0])
            results.append({"path": os.path.join(r"DATA/BRUT/SYNT_LABELED_AUDIO", temp_audio_path), "transcription": transcript})
        df = pd.DataFrame(results)
        return df
    
def main():
    working_path = r"/mnt/f/WOLOF/SPEECH_TO_TEXT/DATA/BRUT/SYNT_LABELED_AUDIO"
    os.chdir(working_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    transcriber = Transcriber(device)
    audios_path = r"/mnt/f/WOLOF/SPEECH_TO_TEXT/DATA/BRUT/WOLOF_AUDIO/audio"
    
    processed_files_path = os.path.join(working_path, "processed_files.json")
    if os.path.exists(processed_files_path):
        with open(processed_files_path, "r") as f:
            processed_files = json.load(f)
    else:
        processed_files = []

    list_audio = os.listdir(audios_path)

    for file_audio in list_audio:
        if file_audio not in processed_files:
            d = transcriber.process_file(audios_path, file_audio)
            df = pd.read_csv(os.path.join(working_path, "transcriptions.csv"))
            df = pd.concat([df, d])
            df.to_csv(os.path.join(working_path, "transcriptions.csv"), index=False)
            processed_files.append(file_audio)
            with open(processed_files_path, "w") as f:
                json.dump(processed_files, f)

if __name__ == "__main__":
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    main()