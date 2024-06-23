from PIL import Image, ImageDraw, ImageFont
import os

def find_next_image(image_folder, frame_num):
    while True:
        image_name = f"frame{frame_num:06}.png"  # Format your frame number to 'frame000000'
        image_path = os.path.join(image_folder, image_name)
        if os.path.exists(image_path):
            return image_path
        frame_num += 1

def plot_frames(image_folder, initial_frame, last_frame, jump_length, output_image_path):
    images = []

    # Load images based on the initial frame, last frame, and jump length
    frame_num = initial_frame
    while frame_num <= last_frame:
        image_path = find_next_image(image_folder, frame_num)
        
        try:
            frame_actual_num = int(image_path.split('frame')[-1].split('.png')[0])
            image = Image.open(image_path)
            images.append((frame_actual_num, image))
        except FileNotFoundError:
            print(f"Image '{image_path}' not found. Skipping frame {frame_num}.")
            continue
        
        frame_num += jump_length

    if not images:
        print("No images to plot. Exiting function.")
        return

    # Assuming all images are of the same size
    frame_width, frame_height = images[0][1].size
    total_width = frame_width * len(images)
    total_height = frame_height + 50  # Adding 50 pixels for frame number labels

    # Create a new image
    result_image = Image.new("RGB", (total_width, total_height))
    draw = ImageDraw.Draw(result_image)

    try:
        # Load a truetype or opentype font file with a larger size
        font = ImageFont.truetype("arial.ttf", 40)
    except IOError:
        # If the truetype font is not available, fall back on the default, but larger
        font = ImageFont.load_default()

    # Paste images and draw frame numbers
    x_offset = 0
    for frame_num, img in images:
        result_image.paste(img, (x_offset, 0))
        draw.text((x_offset + frame_width // 2, frame_height + 5), f"Frame {frame_num}", fill=(255, 255, 255), font=font, anchor="mm")
        x_offset += frame_width
    
    result_image.save(output_image_path)
    print(f"Output image saved as '{output_image_path}'")

# Example usage
plot_frames(image_folder='./evaluate_result/bailando_result/imgs', initial_frame=3000, last_frame=3720, jump_length=120, output_image_path='./evaluate_result/bailando_result/imgs_output.png')