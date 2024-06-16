import csv
import json
import os
from pydub import AudioSegment
import pandas as pd

def process_audio_csv(input_csv_path, output_json_path, output_csv_path):
    audio_data = []
    total_duration = 0
    
    # Read the CSV file
    with open(input_csv_path, newline='', encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            transcription = row['transcription']
            audio_path = row['path']
            new_row = {}
            new_row["transcription"] = row["transcription"]

            # Filter out transcriptions with three words or fewer
            if len(transcription.split()) >= 3:
                # Calculate the duration of the audio file
                audio_path = audio_path.replace("/", "\\")
                audio = AudioSegment.from_file(os.path.join(r"F:\WOLOF\SPEECH_TO_TEXT", audio_path))
                duration = len(audio) / 1000.0  # Convert duration to seconds
                
                new_row['duration'] = duration
                new_row["path"] = audio_path
                audio_data.append(new_row)
                total_duration += duration
            else:
                os.remove(audio_path)
    
    # Calculate the number of audios and mean duration
    num_audios = len(audio_data)
    mean_duration = total_duration / num_audios if num_audios > 0 else 0
    df = pd.DataFrame(audio_data)
    df.to_csv(output_csv_path, index=False)
    
    # Create the result JSON
    result = {
        'number_of_audios': num_audios,
        'total_duration': total_duration,
        'mean_duration': mean_duration
    }
    
    # Write the result to the JSON file
    with open(output_json_path, 'w') as jsonfile:
        json.dump(result, jsonfile, indent=4)
    
    return result


def main():
    os.chdir(r"F:\WOLOF\SPEECH_TO_TEXT")
    input_csv_path = r"DATA\BRUT\SYNT_LABELED_AUDIO\transcriptions.csv"
    output_json_path = r"DATA\CLEANED\SYNT_LABELED_AUDIO\result.json"
    output_csv_path = r"DATA\CLEANED\SYNT_LABELED_AUDIO\filtered_synthetized_data.csv"
    result = process_audio_csv(input_csv_path, output_json_path, output_csv_path)
    print(result)

if __name__ == "__main__":
    main()