import os
import numpy as np
import glob

def concatenate_slices(input_folder, output_folder):
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate through each song directory inside the input folder
    for song_dir in os.listdir(input_folder):
        song_path = os.path.join(input_folder, song_dir)
        if not os.path.isdir(song_path):
            continue

        # List all slice files in the song directory
        slice_files = sorted(glob.glob(os.path.join(song_path, f"{song_dir}_slice*.npy")))
        slices = []

        # Load each slice and append to the slices list
        for slice_file in slice_files:
            slice_data = np.load(slice_file)
            slices.append(slice_data)

        # Concatenate all slices into a single array
        concatenated_data = np.concatenate(slices)

        # Define the output file path
        output_file = os.path.join(output_folder, f"{song_dir}.npy")

        # Save the concatenated array to the output file
        np.save(output_file, concatenated_data)

        print(f"Saved concatenated file for {song_dir} to {output_file}")

if __name__ == "__main__":
    input_folder = "custom_input\\feature_sliced"
    output_folder = "custom_input\\feature"
    concatenate_slices(input_folder, output_folder)