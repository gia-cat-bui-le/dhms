from PIL import Image, ImageDraw, ImageFont
import os

def find_next_image(image_folder, frame_num):
    while True:
        image_name = f"frame_{frame_num:04}.png"
        image_path = os.path.join(image_folder, image_name)
        if os.path.exists(image_path):
            return image_path
        frame_num += 1

def plot_frames(image_folder, initial_frame, last_frame, jump_length, output_image_path):
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
        # draw.text(text_position, f"Frame {frame_num}", fill=(255, 255, 255), font=font, anchor="mm")

        x_offset += frame_width
        if (i + 1) % num_columns == 0:
            x_offset = 0
            y_offset += frame_height
    
    result_image.save(output_image_path)
    print(f"Output image saved as '{output_image_path}'")

# Example usage
plot_frames(image_folder='./frames/114', initial_frame=1, last_frame=310, jump_length=30, output_image_path='./frames/output.png')