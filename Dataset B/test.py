import os
import pandas as pd
import pandas as pd

def add_double_quotes(csv_file):
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_file)

    # Wrap each cell's content with double quotes if there are no double quotes already present
    df = df.applymap(lambda x: f'"{x}"' if '"' not in str(x) else x)

    # Write the modified DataFrame back to the same CSV file
    df.to_csv(csv_file, index=False)

# Usage example:
csv_file_path = '/workspace/Project_Final/MusicSeg/Dataset B/filename_mapping.csv'
add_double_quotes(csv_file_path)
# Function to rename files and create CSV mapping
def rename_files(directory):
    # Initialize dataframe to store mapping
    mapping_df = pd.DataFrame(columns=['initial_filename', 'new_filename'])

    # Initialize counter for new filename
    counter = 0

    # Iterate through files in the directory
    for filename in sorted(os.listdir(directory)):
        print(filename)
        if filename.endswith('.mp3'):
            # Generate new filename
            new_filename = '{:03d}.mp3'.format(counter)

            # Rename the file
            os.rename(os.path.join(directory, filename), os.path.join(directory, new_filename))

            # Add mapping to dataframe
            mapping_df = mapping_df.append({'initial_filename': filename, 'new_filename': new_filename}, ignore_index=True)

            # Increment counter
            counter += 1

    # Save dataframe to CSV
    mapping_df.to_csv('filename_mapping.csv', index=False)

# Directory containing the MP3 files
directory = '/workspace/Project_Final/MusicSeg/Dataset B/Raw/Songs/Songs'

# Call the function to rename files and create CSV mapping
rename_files(directory)
