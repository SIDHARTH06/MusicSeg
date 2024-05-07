textgrid_dir = '/workspace/Project_Final/MusicSeg/Dataset B/Raw/Songs/TextGrid Files'
json_dir = '/workspace/Project_Final/MusicSeg/Dataset B/Labels/'
import os
if not os.path.exists(json_dir):
    os.makedirs(json_dir)
import pandas as pd
import textgrid
for textgrid_file in os.listdir(textgrid_dir):
    textgrid_file_path = os.path.join(textgrid_dir, textgrid_file)

    try:
        tgrid = textgrid.read_textgrid(textgrid_file_path)
        json_file_path = os.path.join(json_dir, f"{textgrid_file[:-9]}.json")
        pd.DataFrame(tgrid).to_json(json_file_path)
    except Exception as e:
        print(e)
