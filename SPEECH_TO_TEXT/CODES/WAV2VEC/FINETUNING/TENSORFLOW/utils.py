import pandas as pd
import os


def extract_all_chars():
    all_text = ""
    directory = "/kaggle/input/wolof-speech2text/alffa/alffa"
    file_name = "alffa_clean_df.csv"
    with open(os.path.join(directory, file_name), 'rb') as f:
        batch = pd.read_csv(f)
    for k in range(len(batch)):
        all_text += " " + batch['transcription'][k]

    vocab = list(set(all_text))
    return {"vocab": [vocab], "all_text": [all_text]}


def read_wav_file(file_path, REQUIRED_SAMPLE_RATE = 16000):
  with open(file_path, "rb") as f:
      audio, sample_rate = sf.read(f)
  if sample_rate != REQUIRED_SAMPLE_RATE:
      raise ValueError(
          f"sample rate (={sample_rate}) of your files must be {REQUIRED_SAMPLE_RATE}"
      )
  file_id = os.path.split(file_path)[-1][:-len(".wav")]
  return {file_id: audio}


def text_preprocess(row):
    text = row.get('transcription')
    samples = text.split("\n")
    samples = {row.get('filename')[:-len(".wav")]: " ".join(s.split()[1:]) for s in samples if len(s.split()) > 2}
    return samples


def fetch_sound_text_mapping(data_dir, text_file_df):
  all_files = os.listdir(data_dir)

  wav_files = [os.path.join(data_dir, f) for f in all_files if f.endswith(".wav")]
  aux = text_file_df.apply(lambda row: text_preprocess(row), axis = 1)

  txt_samples = {}
  for (_, text_sample) in aux.items():
    txt_samples.update(text_sample)

  speech_samples = {}
  for f in wav_files:
    speech_samples.update(read_wav_file(f))

  assert len(txt_samples) == len(speech_samples)

  samples = [(speech_samples[file_id], txt_samples[file_id]) for file_id in speech_samples.keys() if len(speech_samples[file_id]) < AUDIO_MAXLEN]
  return samples