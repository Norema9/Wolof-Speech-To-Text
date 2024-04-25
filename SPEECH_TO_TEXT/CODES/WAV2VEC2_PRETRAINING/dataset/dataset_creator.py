import os
import argparse
import toml
import pandas as pd
from pydub import AudioSegment
from sklearn.model_selection import train_test_split
import webrtcvad
import wave
import json
from tqdm import tqdm

def contains_voice(file_path):
    # Set aggressiveness mode, which is an integer between 0 and 3. 
    # 0 is the least aggressive about filtering out non-speech, 3 is the most aggressive.
    vad = webrtcvad.Vad(3)

    # Open the wave file
    audio = wave.open(file_path, 'rb')

    # Read frames from the audio file
    frames = []
    frame_duration = 30 
    for _ in range(int(audio.getnframes() / (audio.getframerate() * frame_duration / 1000))):
        frames.append(audio.readframes(audio.getframerate() * frame_duration // 1000))

    # Check if any frame contains voice
    return any(vad.is_speech(frame, audio.getframerate()) for frame in frames)


# List all mp3 files in a directory
def list_mp3_files(directory):
    return [f for f in os.listdir(directory) if f.endswith('.wav')]

# Compute duration of audio file
def compute_duration_and_rate(file_path):
    audio = AudioSegment.from_file(file_path)

    duration = len(audio) / 1000  # Duration in seconds
    rate = audio.frame_rate  # Sample rate in Hz
    
    return duration, rate



def main(config):

    audio_brut_dir = config["meta"]["audio_dir"]
    dataset_dir = config["meta"]["dataset_dir"]
    sep = config['meta']['separator']
    seed = config['meta']['seed']
    total_duration = 0
    number_of_audio = 0
    problem_with_this_audio = []

    mp3_files = list_mp3_files(audio_brut_dir)
    data = []
    for file in tqdm(mp3_files):
        file_path = os.path.join(audio_brut_dir, file)
        try:
            if contains_voice(file_path):
                duration, rate = compute_duration_and_rate(file_path)
                total_duration += duration
                data.append({'name': file, 'path': file_path, 'duration': duration, 'rate': rate})
                number_of_audio += 1
        except:
            problem_with_this_audio.append({"name": file})
            
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(dataset_dir, 'dataset_clean.csv'), sep = sep, index = False)
    
    df_problem = pd.DataFrame(problem_with_this_audio)
    df_problem.to_csv(os.path.join(dataset_dir, 'problem_with_this_audio_file.csv'), sep = sep, index = False)

    # Split the data into training and validation sets (80% train, 20% validation)
    X_train, X_val = train_test_split(df, test_size=0.1, random_state = seed)

    X_train.to_csv(os.path.join(dataset_dir, 'train_dataset.csv'), sep = sep, index = False)
    X_val.to_csv(os.path.join(dataset_dir, 'validation_dataset.csv'), sep = sep, index = False)
    
    train_duration = X_train["duration"].sum()
    validation_duration = X_val["duration"].sum()
    train_number_of_audio = len(X_train)
    validation_number_of_audio = len(X_val)
    
    info = {"totale_dration": total_duration, 
            "number_of_audio": number_of_audio, 
            "train_duration": train_duration, 
            "validation_duration": validation_duration,
            "train_number_of_audio": train_number_of_audio,
            "validation_number_of_audio": validation_number_of_audio}
    
    # Write the total_duration variable to a JSON file
    with open(os.path.join(dataset_dir, 'dataset_info.json'), 'w') as json_file:
        json.dump(info, json_file)

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='ASR TRAIN ARGS')
    args.add_argument('-c', '--config', required=True, type=str,
                      help='config file path (default: None)')      
    
    args = args.parse_args()
    config = toml.load(args.config)
    main(config)
    
    
