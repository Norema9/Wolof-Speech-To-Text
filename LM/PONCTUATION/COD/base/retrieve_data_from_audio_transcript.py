import pandas as pd
import glob, os


tot = []
directory = r"D:\MARONE\WOLOF\SPEECH_TO_TEXT\DATA\CLEANED\WOLOF_AUDIO_TRANS\zz_total_cleaned"

os.chdir(directory)
for file in glob.glob("*.csv"):
    tot.append(pd.read_csv(file)["transcription"])
    
tot_transcript_df = pd.concat(tot, axis = 0, ignore_index = True)

tot_transcript_df.to_csv(r"D:\MARONE\WOLOF\LM\PONCTUATION\DATA\data_brut\tot_transcription_data.csv", index=False)