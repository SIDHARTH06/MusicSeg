import os
import pandas as pd
import re
# Step 1: Read the CSV file into a pandas DataFrame
mapping_df = pd.read_csv('/workspace/Project_Final/MusicSeg/Dataset B/filename_mapping.csv')
def remove_special_chars(string):
    # Define a regular expression pattern to match any character that is not alphanumeric or a period
    pattern = r'[^a-zA-Z0-9.]'
    # Use the sub() function from the re module to replace all matches of the pattern with an empty string
    return re.sub(pattern, '', string)
# Step 2: Iterate through the textgrid files in the directory
textgrid_directory = '/workspace/Project_Final/MusicSeg/Dataset B/Raw/Songs/TextGrid Files'
for filename in os.listdir(textgrid_directory):
    if filename.endswith('.TextGrid'):
        # Step 3: Extract initial filename without extension
        initial_filename = os.path.splitext(filename)[0] + '.mp3'
        initial_filename = remove_special_chars(initial_filename)
        # Step 4: Check if the initial filename exists in the mapping DataFrame
        mapping_row = mapping_df[mapping_df['initialfilename'] == initial_filename]
        if not mapping_row.empty:
            # Get the new filename from the mapping DataFrame
            new_filename = mapping_row.iloc[0]['newfilename']
            
            # Construct the new filename for the textgrid file
            new_textgrid_filename = new_filename.rsplit('.', 1)[0] + '.TextGrid'
            
            # Step 5: Rename the textgrid file
            os.rename(os.path.join(textgrid_directory, filename), os.path.join(textgrid_directory, new_textgrid_filename))
            # print(f"Renamed {filename} to {new_textgrid_filename}")
        else:
            print(filename)
