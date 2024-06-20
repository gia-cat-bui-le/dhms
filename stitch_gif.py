from PIL import Image, ImageSequence
import os

def combine_gifs(gif1_path, gif2_path, output_path):
    # Open the two GIFs
    gif1 = Image.open(gif1_path)
    gif2 = Image.open(gif2_path)

    # Create a list to hold the individual frames of the final GIF
    combined_frames = []

    # Iterate through the frames of the first GIF
    for frame1 in ImageSequence.Iterator(gif1):
        # Seek to the corresponding frame in the second GIF
        try:
            gif2.seek(gif1.tell())
            frame2 = gif2.copy()
        except EOFError:
            break

        # Create a new image with width=gif1+gif2 and height=max(gif1, gif2)
        new_frame = Image.new('RGBA', (frame1.width + frame2.width, max(frame1.height, frame2.height)))

        # Paste the two frames side by side
        new_frame.paste(frame1, (0, 0))
        new_frame.paste(frame2, (frame1.width, 0))

        # Append the combined frame to the list
        combined_frames.append(new_frame)

    # Save the combined frames as a new GIF
    combined_frames[0].save(output_path, save_all=True, append_images=combined_frames[1:], loop=0, duration=gif1.info['duration'])

def combine_gifs_from_folders(my_folder, other_folders, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    my_gifs = [os.path.join(my_folder, f) for f in os.listdir(my_folder) if f.lower().endswith('.gif')]

    for other_folder in other_folders:
        other_gifs = [os.path.join(other_folder, f) for f in os.listdir(other_folder) if f.lower().endswith('.gif')]
        combined_folder_name = os.path.basename(my_folder) + '_' + os.path.basename(other_folder)
        combined_folder_path = os.path.join(output_folder, combined_folder_name)

        if not os.path.exists(combined_folder_path):
            os.makedirs(combined_folder_path)

        for i, my_gif in enumerate(my_gifs):
            for j, other_gif in enumerate(other_gifs):
                output_gif_path = os.path.join(combined_folder_path, f'combined_{i+1}_{j+1}.gif')
                combine_gifs(my_gif, other_gif, output_gif_path)

# Example usage
my_folder = 'visualize_result\DHMS\guidance-2.5'
other_folders = ['visualize_result\Bailando', 'visualize_result\FACT', 'visualize_result\EDGE']
output_folder = 'visualize_result\combine_result\guidance-2.5'

combine_gifs_from_folders(my_folder, other_folders, output_folder)