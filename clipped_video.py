from moviepy.video.io.VideoFileClip import VideoFileClip

def clip_video(input_video_path, output_video_path, start_time, end_time):
    """
    Clips a video from start_time to end_time and saves the result to output_video_path.

    :param input_video_path: Path to the input video file.
    :param output_video_path: Path to save the clipped video.
    :param start_time: Start time in seconds.
    :param end_time: End time in seconds.
    """
    # Load the video
    video = VideoFileClip(input_video_path)

    # Clip the video
    clipped_video = video.subclip(start_time, end_time)

    # Write the result to a file
    clipped_video.write_videofile(output_video_path, codec="libx264")

# Example usage
input_video_path = "visualize_result\chosen_video\edge\\035.mp4"
output_video_path = "visualize_result\chosen_video\clipped\\035_edge.mp4"
start_time = 65  # Start time in seconds
end_time = 95    # End time in seconds

clip_video(input_video_path, output_video_path, start_time, end_time)