import multiprocessing
import pandas as pd
from time import time
from pose_estimation.scripts import pose_api as pose_api
from draw_skeleton import draw_skeleton
import skvideo.io as skio
import sys
import os

api = pose_api.PoseApi()



def load_video_lookup(path):
    return pd.read_csv(path, converters={'video_id': str})

def get_video_data(video_id):
    return draw_skeleton(video_id)

def get_video_ids(df):
    return df['video_id'].values

def process_videos(video_ids):
    num_to_process = len(video_ids)
    num_processed = 0
    running_duration = 0
    for video_id in video_ids:
        start_time = time()
        video_id = video_id.replace(r".mp4", "")
        store_data = get_video_data(video_id).tolist()
        
        skio.vwrite(f"dataset/processed/{video_id}.mp4", store_data)
        
        
        num_to_process -= 1
        num_processed += 1
        
        running_duration += time() - start_time
        
        sys.stdout.flush()
        sys.stdout.write(f"\r{video_id} has been processed, {((running_duration / num_processed) * num_to_process) // 60} minutes left")
        

if __name__ == "__main__":
    if not os.path.isdir("dataset/processed/"):
        os.mkdir("dataset/processed/")
    # After lookup creation
    video_lookup = load_video_lookup("dataset/LSA64_lookup.csv")
    video_ids = get_video_ids(video_lookup)
    
    # Split the video ids into several parts, to speed up pre processing
    split_video_ids = [
        video_ids[:640], 
        video_ids[640:1280], 
        video_ids[1280:1920],
        video_ids[1920:]
        ]
    
    with multiprocessing.Pool() as pool:
        pool.map(process_videos, split_video_ids)
    
    
        
    print("Successfully saved")