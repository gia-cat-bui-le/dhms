from PIL import Image, ImageDraw, ImageFont
import os

def find_next_image(image_folder, frame_num):
    while True:
        image_name = f"inpainting 20_frame_{frame_num:04}.png"
        image_path = os.path.join(image_folder, image_name)
        if os.path.exists(image_path):
            return image_path
        frame_num += 1

def find_images(initial_frame, last_frame, image_folder, jump_length):
    images = []
    crop_box = (200, 0, 500, 500)  # Define the crop box (left, upper, right, lower)

    # Load images based on the initial frame, last frame, and jump length
    frame_num = initial_frame
    while frame_num <= last_frame:
        image_path = find_next_image(image_folder, frame_num)
        
        try:
            frame_actual_num = int(image_path.split('frame_')[-1].split('.png')[0])
            image = Image.open(image_path)
            image = image.crop(crop_box)  # Crop the image
            images.append((frame_actual_num, image))
        except FileNotFoundError:
            print(f"Image '{image_path}' not found. Skipping frame {frame_num}.")
            continue
        
        frame_num += jump_length

    if not images:
        print("No images to plot. Exiting function.")
        return
    
    return images

def plot_frames(output_image_path):
    images = []
    crop_box = (200, 0, 500, 500)  # Define the crop box (left, upper, right, lower)

    # Load images based on the initial frame, last frame, and jump length
    images = find_images(image_folder='./frames\inpainting 20', initial_frame=165, last_frame=169, jump_length=1)

    # Number of columns and rows in the grid
    num_columns = 5
    num_rows = 2

    # Assuming all images are of the same size
    frame_width, frame_height = images[0][1].size
    total_width = frame_width * num_columns
    total_height = frame_height * num_rows

    # Create a new image
    result_image = Image.new("RGB", (total_width, total_height))
    draw = ImageDraw.Draw(result_image)

    try:
        # Load a truetype or opentype font file with a larger size
        font = ImageFont.truetype("arial.ttf", 40)
    except IOError:
        # If the truetype font is not available, fall back on the default, but larger
        font = ImageFont.load_default()

    # Paste images in grid and draw frame numbers
    x_offset = 0
    y_offset = 0
    for i, (frame_num, img) in enumerate(images):
        result_image.paste(img, (x_offset, y_offset))

        # Draw the frame number below the image
        text_position = (x_offset + frame_width // 2, y_offset + frame_height + 5)
        draw.text(text_position, f"Frame {frame_num}", fill=(255, 255, 255), font=font, anchor="mm")

        x_offset += frame_width
        if (i + 1) % num_columns == 0:
            x_offset = 0
            y_offset += frame_height
    
    images = find_images(image_folder='./frames\inpainting 40', initial_frame=165, last_frame=169, jump_length=1)
    
    result_image.save(output_image_path)
    print(f"Output image saved as '{output_image_path}'")

# Example usage
#plot_frames(image_folder='./frames/114', initial_frame=1, last_frame=310, jump_length=30, output_image_path='./frames/output.png')

import os
import subprocess

def extract_frames(video_path, output_folder):
    """
    Extract frames from a video and save them to a specified output folder.

    Args:
    - video_path: Path to the video file.
    - output_folder: Folder where the frames will be saved.
    """
    # Create the output folder if it does not exist
    os.makedirs(output_folder, exist_ok=True)

    # Extract the base name of the video file without extension
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    # Construct the ffmpeg command
    cmd = [
        'ffmpeg',
        '-i', video_path,
        '-vf', 'fps=30',  # You can change fps=1 to a different frame rate if needed
        os.path.join(output_folder, f"{video_name}_frame_%04d.png")
    ]

    # Execute the command
    subprocess.run(cmd, check=True)

def main():
    # Path to the directory containing the videos
    video_directory = 'frames'

    # Iterate over all video files in the directory
    for video_file in os.listdir(video_directory):
        if video_file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):  # Add more video extensions if needed
            video_path = os.path.join(video_directory, video_file)
            output_folder = os.path.join(video_directory, os.path.splitext(video_file)[0])

            # Extract frames from the video
            extract_frames(video_path, output_folder)
            print(f"Frames from {video_file} saved to folder {output_folder}")

if __name__ == "__main__":
    # main()
    plot_frames(output_image_path='./frames/inpainting-ablation.png')