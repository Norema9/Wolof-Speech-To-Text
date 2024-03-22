import librosa

def get_audio_duration(input_path, path_audio):
    # Load the audio file
    audio, sr = librosa.load(path_audio + input_path + '.wav', sr = None)

    # Calculate the duration of the original audio
    audio_duration = librosa.get_duration(y=audio, sr=sr)

    return audio_duration