import os
import json
import subprocess
import numpy as np
import math
from concurrent.futures import ProcessPoolExecutor
import cv2

def get_video_resolution_cv2(input_path):
    # Open video file
    video = cv2.VideoCapture(input_path)
    # Get width and height
    original_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video.release()
    return original_width, original_height

def ensure_dir(directory):
    """
    Ensure the given directory exists, create it if it does not.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

def extract_frames_single(src_path, src_name, des_path, img_path):
    """
    Extract frames from a single video file and save them to the output folder.
    """
    ensure_dir(os.path.join(des_path, img_path))

    video_file = f"{src_name}.mp4"
    true_count = 0
    if video_file.endswith(('.mp4', '.avi')):
        video_path = os.path.join(src_path, video_file)
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        count = 0
        true_count = 1
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            interval = int(fps)  # Extract one frame per second
            if count % interval == 0:
                frame_name = f"{os.path.splitext(video_file)[0]}_frame_{true_count}.jpg"
                cv2.imwrite(os.path.join(des_path, img_path, frame_name), frame)
                true_count += 1
            count += 1
        cap.release()
    return true_count

def extract_frames(src_path, des_path):
    """
    Extract frames from all videos in the given folder and save them to the output folder.
    """
    ensure_dir(des_path)

    for video_file in os.listdir(src_path):
        if video_file.endswith(('.mp4', '.avi')):
            extract_frames_single(src_path, os.path.splitext(video_file)[0], des_path, '')

def set_video_frame(src_path, src_name, des_path, des_name, frame_number, frame_begin_number=0):
    """
    Split a video from a specified frame and save it to another folder.
    """
    video = os.path.join(src_path, f"{src_name}.mp4")
    cap = cv2.VideoCapture(video)
    frame_count = int(cap.get(cv2.CAP_PROP_FPS))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    videowriter = cv2.VideoWriter(os.path.join(des_path, f"{des_name}.mp4"), cv2.VideoWriter_fourcc(*'mp4v'),
                                  frame_count, (frame_width, frame_height))

    count = -frame_begin_number
    success, img = cap.read()
    while success:
        if count >= 0:
            videowriter.write(img)
        if count == frame_number - 1:
            break
        success, img = cap.read()
        count += 1


def set_video_frame_rate(src_path, src_name, des_path, des_name, fps_list):
    """
    Adjust the frame rate of a video and save to a new file.
    """
    video = os.path.join(src_path, f"{src_name}.mp4")
    cap = cv2.VideoCapture(video)

    if not cap.isOpened():
        print(f"Error opening video file {video}")
        return

    src_fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    for target_fps in fps_list:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        if target_fps <= 0:
            print("Target FPS must be greater than 0.")
            continue

        frame_interval = src_fps / target_fps
        out_video = os.path.join(des_path, f"{des_name}_fps_{target_fps}.mp4")
        videowriter = cv2.VideoWriter(out_video, cv2.VideoWriter_fourcc(*'mp4v'), target_fps,
                                      (frame_width, frame_height))

        success, img = cap.read()
        frame_id = 0

        while success:
            if frame_id % frame_interval < 1:
                videowriter.write(img)
            success, img = cap.read()
            frame_id += 1

        videowriter.release()

    cap.release()

def check_file_status(filename):
    """
    Check if the file exists and is not empty.
    """
    if not os.path.exists(filename) or os.path.getsize(filename) == 0:
        return 0
    return 1

def load_config(config_path='motivation_config.json'):
    """
    Load a configuration file.
    """
    with open(config_path, 'r') as config_file:
        return json.load(config_file)

# 
def get_video_duration_ffprobe(video_path):
    """
    Use ffprobe to get the duration of the video in seconds and round it to the nearest second.
    """
    command = [
        'ffprobe', '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        video_path
    ]
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        duration = float(result.stdout.strip())
        return math.floor(duration)  # Round up to the nearest second
    except Exception as e:
        print(f"Error getting video duration: {e}")
        return 0

## 
def set_video_segment_ffmpeg(src_path, src_name, des_path, segment_duration_minutes, video_counter):
    """
    Split a video into segments of specified duration using ffmpeg.
    """
    ensure_dir(des_path)
    segment_duration = segment_duration_minutes * 60  # convert minutes to seconds
    input_path = os.path.join(src_path, f"{src_name}.mp4")

    # Get the total duration of the video
    total_duration = get_video_duration_ffprobe(input_path)

    # Calculate number of full segments
    num_full_segments = int(total_duration // segment_duration)

    for i in range(num_full_segments):
        output_file = os.path.join(des_path, f"video_{video_counter}.mp4")
        start_time = i * segment_duration
        command = [
            'ffmpeg', '-ss', str(start_time), '-i', input_path, '-t', str(segment_duration),
            '-c:v', 'copy', '-an', output_file, '-y'
        ]
        subprocess.run(command, check=True)
        video_counter += 1

    # Check if there's a remaining segment that's less than the full duration but more than 10 minutes
    remaining_time = total_duration - (num_full_segments * segment_duration)
    if remaining_time >= 600:  # 10 minutes = 600 seconds
        output_file = os.path.join(des_path, f"video_{video_counter}.mp4")
        start_time = num_full_segments * segment_duration
        command = [
            'ffmpeg', '-i', input_path, '-ss', str(start_time), '-c:v', 'copy', '-an', output_file, '-y'
        ]
        subprocess.run(command, check=True)
        video_counter += 1

    return video_counter

def segment_all_videos_in_folder(input_folder, output_folder, segment_duration_minutes):
    """
    Segment all videos in the input folder and save them in the output folder.
    """
    ensure_dir(output_folder)
    video_counter = 1  # Start numbering segments from 1

    for video_file in os.listdir(input_folder):
        if video_file.endswith(('.mp4', '.avi')):
            video_path = os.path.join(input_folder, video_file)
            video_length_minutes = get_video_duration_ffprobe(video_path) / 60

            # Skip videos less than 10 minutes long
            if video_length_minutes < 10:
                print(f"Skipping video {video_file} as it is too short to segment.")
                continue

            # Segment the video and update the video_counter
            video_counter = set_video_segment_ffmpeg(input_folder, os.path.splitext(video_file)[0], output_folder,
                                                     segment_duration_minutes, video_counter)

## 
def set_video_single_segment_ffmpeg(src_path, src_name, des_path, segment_duration_minutes, video_counter):
    """
    Split a video into segments of specified duration using ffmpeg.
    """
    ensure_dir(des_path)
    segment_duration = 60*segment_duration_minutes  # convert minutes to seconds
    input_path = os.path.join(src_path, f"{src_name}.mp4")
    # Get the total duration of the video
    total_duration = get_video_duration_ffprobe(input_path)
    output_file = os.path.join(des_path, f"video_{video_counter}.mp4")
    start_time = 0
    command = [
        'ffmpeg', '-ss', str(start_time), '-i', input_path, '-t', str(segment_duration),
        '-c:v', 'copy', '-an', output_file, '-y'
    ]
    subprocess.run(command, check=True)

def single_segment_all_videos_in_folder(input_folder, output_folder, segment_duration_minutes):
    """
    Segment all videos in the input folder and save them in the output folder.
    """
    ensure_dir(output_folder)
    for i in range(1,44):
        video_file = 'video_'+str(i)+'.mp4'
        video_path = os.path.join(input_folder, video_file)
        video_length_minutes = get_video_duration_ffprobe(video_path) / 60
        set_video_single_segment_ffmpeg(input_folder, os.path.splitext(video_file)[0], output_folder,
                                                 segment_duration_minutes, i)


def base_extract_video_segment_size_and_keyframe(video_path, output_folder, segment_range):
    ensure_dir(output_folder)
    total_duration = get_video_duration_ffprobe(video_path)
    print(total_duration)
    if total_duration == 0:
        print(f"Cannot get duration for video: {video_path}")
        return []
    segment_sizes = []

    for second in range(min(int(total_duration), segment_range)):
        segment_output = os.path.join(output_folder, f"segment_{second + 1}.mp4")
        segment_duration = 1 if second < int(total_duration) - 1 else total_duration - second
        if segment_duration < 1:
            print(f"Remaining segment is less than 1 second. Stopping at second {second}.")
            break
        command = [
            'ffmpeg', '-ss', str(second), '-i', video_path, '-t', str(segment_duration),
            '-c:v', 'libx264', '-an', segment_output, '-y'
        ]
        try:
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            print(f"Created segment: {segment_output}")
        except subprocess.CalledProcessError as e:
            print(f"Error extracting segment at {second} second: {e.stderr}")
            continue
        if os.path.getsize(segment_output) == 0:
            print(f"Warning: Segment {segment_output} is empty, skipping.")
            segment_sizes.append(0)
            continue
        try:
            segment_size = os.path.getsize(segment_output)
            segment_sizes.append(segment_size)
        except OSError as e:
            print(f"Error getting file size for {segment_output}: {e}")
            segment_sizes.append(0)
        keyframe_output = os.path.join(output_folder, f"frame_{second + 1}.jpg")
        command = [
            'ffmpeg', '-i', segment_output, '-vf', 'select=eq(pict_type\,I)', '-vsync', 'vfr', '-frames:v', '1',
            keyframe_output, '-y'
        ]
        try:
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            print(f"Extracted keyframe: {keyframe_output}")
        except subprocess.CalledProcessError as e:
            print(f"Error extracting keyframe for segment {second + 1}: {e.stderr}")
            continue

    return segment_sizes

def base_delete_all_videos(input_folder, frame_numbers, begin, end):
    video_files = [f"video_{i}.mp4" for i in range(begin, end)]
    for video_file in video_files:
        if video_file.endswith(('.mp4', '.avi')):
            output_folder = os.path.join(input_folder, f"{os.path.splitext(video_file)[0]}_segments") # 
            for i in range(frame_numbers):
                segment_output = os.path.join(output_folder, f"segment_{i + 1}.mp4")
                try:
                    os.remove(segment_output)
                except OSError as e:
                    print(f"Error deleting file {segment_output}: {e}")


def base_process_all_videos(input_folder,frame_numbers, begin, end):
    video_files = [f"video_{i}.mp4" for i in range(begin, end)]
    for video_file in video_files:
        # if video_file == "video_44.mp4" or video_file == "video_45.mp4":
        if video_file.endswith(('.mp4', '.avi')):
            video_path = os.path.join(input_folder, video_file)
            output_folder = os.path.join(input_folder, f"{os.path.splitext(video_file)[0]}_segments") # 
            # Extract segment sizes and keyframes
            segment_sizes = base_extract_video_segment_size_and_keyframe(video_path, output_folder,frame_numbers)
            # Save the segment sizes array to a file
            sizes_output_path = os.path.join(output_folder, 'segment_sizes.npy')
            np.save(sizes_output_path, np.array(segment_sizes))

def reencode_single_video(input_path, video_file, output_folder, qp, res_factor):
    """
    Re-encode a single video with a given QP value and resolution factor.
    Extract keyframe and delete the video after processing.
    """
    # Define the output resolution
    width, height = get_video_resolution_cv2(input_path)


    # Calculate the new height and width with res_factor
    new_width = int(width * res_factor)
    new_height = int(height * res_factor)
    # Ensure the height is even
    # restrat = 0
    if new_height % 2 != 0:
        # restrat = 1
        new_height += 1  # Adjust height to be even if it's odd
    if new_width % 2 != 0:
        # restrat = 1
        new_width += 1  # Adjust height to be even if it's odd
    # Define the output resolution string
    rescale_str = f"scale={new_width}:{new_height}"
    # Define the output file path with resolution factor in the filename
    reencoded_file = os.path.join(output_folder,
                                  f"{os.path.splitext(video_file)[0]}_qp{qp}_res{res_factor}.mp4")

    # Re-encode the video with the specified QP value and resolution
    command = [
        'ffmpeg', '-i', input_path, '-c:v', 'libx264', '-qp', str(qp), '-vf', rescale_str, '-an',
        reencoded_file, '-y'
    ]
    try:
        subprocess.run(command, capture_output=True, text=True, check=True)
        reencoded_size = os.path.getsize(reencoded_file)
    except subprocess.CalledProcessError as e:
        print(f"Error re-encoding {video_file} with QP={qp} and resolution factor={res_factor}: {e.stderr}")
        return 0
    except OSError as e:
        print(f"Error getting file size for {reencoded_file}: {e}")
        return 0

    # Extract the first frame (keyframe) of the re-encoded video
    keyframe_output = os.path.join(output_folder,
                                   f"{os.path.splitext(video_file)[0]}_qp{qp}_res{res_factor}_frame.jpg")
    command = [
        'ffmpeg', '-i', reencoded_file, '-vf', 'select=eq(pict_type\\,I)', '-vsync', 'vfr', '-frames:v',
        '1', keyframe_output, '-y'
    ]
    try:
        subprocess.run(command, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error extracting keyframe for {reencoded_file}: {e.stderr}")

    # Delete the re-encoded video after size and keyframe extraction
    try:
        os.remove(reencoded_file)
    except OSError as e:
        print(f"Error deleting file {reencoded_file}: {e}")

    return reencoded_size

def process_video(video_idx, video_file, input_folder, output_folder, qp_values, res_factors):
    """
    Process a single video file for all QP and resolution combinations and return the size data.
    """
    input_path = os.path.join(input_folder, video_file)
    video_size_data = np.zeros((len(qp_values), len(res_factors)))  # Local size data for this video

    for i, qp in enumerate(qp_values):
        for j, res_factor in enumerate(res_factors):
            reencoded_size = reencode_single_video(input_path, video_file, output_folder, qp, res_factor)
            video_size_data[i, j] = reencoded_size

    return video_idx, video_size_data



def reencode_segments_with_qp(input_folder, output_folder, qp_values, res_factors, frame_num):
    """
    Re-encode all videos in the input folder with different QP values and resolution factors in parallel.
    Save their sizes, extract keyframes, and delete the re-encoded videos after size information is saved.
    """
    output_folder = os.path.join(output_folder, 'qp_encode')
    os.makedirs(output_folder, exist_ok=True)

    video_files = [f"segment_{i}.mp4" for i in range(1, frame_num + 1)]
    size_data = np.zeros((len(video_files), len(qp_values), len(res_factors)))

    # Parallel processing using ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = [
            executor.submit(process_video, video_idx, video_file, input_folder, output_folder, qp_values, res_factors)
            for video_idx, video_file in enumerate(video_files)]

        # Collect results from all processes
        for future in futures:
            video_idx, video_size_data = future.result()
            size_data[video_idx] = video_size_data

    # Save size data for all videos, QP values, and resolutions
    sizes_output_path = os.path.join(output_folder, 'segment_sizes.npy')
    np.save(sizes_output_path, size_data)
    print(f"Saved size data to {sizes_output_path}")

def reencode_all_videos_segments(input_folder, qp_values, res_factors, begin, end, frame_num):
    video_files = [f"video_{i}.mp4" for i in range(begin, end)]
    for video_file in video_files:
        if video_file.endswith(('.mp4', '.avi')):
            segment_output_folder = os.path.join(input_folder, f"{os.path.splitext(video_file)[0]}_segments") #
            reencode_segments_with_qp(segment_output_folder, segment_output_folder, qp_values, res_factors, frame_num)

def count_zeros_and_nans(arr):
    #  np.isnan  NaN 
    nan_count = np.isnan(arr).sum()
    #  arr == 0  NaN  0 
    zero_count = np.sum(arr == 0) - nan_count  #  NaN 0
    print(f"Zero count: {zero_count}, NaN count: {nan_count}")
    return zero_count, nan_count


def process_videos_sizes(begin, end, experiment=False):
    all_aa = []
    all_bb = []
    mid_path = "timevary"
    for i in range(begin, end):
        video_size_path = f"E:/dataset/dash_video/{mid_path}/video_{i}_segments/segment_sizes.npy"
        knobs_video_size_path = f"E:/dataset/dash_video/{mid_path}/video_{i}_segments/qp_encode/segment_sizes.npy"
        aa = np.load(video_size_path, allow_pickle=True)
        bb = np.load(knobs_video_size_path, allow_pickle=True)
        print(f"Video {i} - aa shape: {aa.shape}, bb shape: {bb.shape}")
        all_aa.append(aa)
        all_bb.append(bb)
    all_aa = np.stack(all_aa, axis=0)
    all_bb = np.stack(all_bb, axis=0)
    all_bb = np.transpose(all_bb, (0, 2, 3, 1))
    np.save("data/experiment_video_sizes_knobs_segments.npy", all_bb)
    np.save("data/experiment_video_sizes_segments.npy", all_aa)
    # return all_aa, all_bb


def main():
    pass
    # input_folder = r'E:\dataset\dash_video'
    # segment_output_folder = os.path.join(input_folder, 'segment')
    # segment_output_folder = r'E:\code\python_projects\O2MEGA_TEST\source_video_demo'
    # segment_duration_minutes = 10  # Example segment duration in minutes
    # segment_all_videos_in_folder(input_folder, segment_output_folder, segment_duration_minutes)
    # segment_duration_minutes = 5  # Example segment duration in minutes
    # single_segment_all_videos_in_folder(input_folder, segment_output_folder, segment_duration_minutes)
    # QP_list = [26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48]
    # res_list = [1, 0.7, 0.5, 0.3]
    # segment_output_folder = r'E:\dataset\dash_video\timevary'
    # begin = 1
    # end = 44
    # frame_num = 300
    # cost a lot of time!!!
    # base_process_all_videos(segment_output_folder, frame_num, begin, end)
    # reencode_all_videos_segments(segment_output_folder, QP_list, res_list, begin, end, frame_num)
    # base_delete_all_videos(segment_output_folder, 300 ,begin, end)
    # process_videos_sizes(begin, end,True)


if __name__ == "__main__":
    main()
