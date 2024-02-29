import os
import pandas as pd

if __name__ == "__main__":

    # Creating DataFrame
    json_df = pd.read_json('dataset\WLASL\WLASL_v0.3.json')

    df_data = {
        'gloss': [],
        'bbox': [],
        'fps': [],
        'frame_end': [],
        'frame_start': [],
        'instance_id': [],
        'signer_id': [],
        'source': [],
        'split': [],
        'url': [],
        'variation_id':[],
        'video_id': []
        }



    for _, row in json_df.iterrows():
        gloss = row['gloss']
        instances = row['instances']
        
        
        for instance in instances:
            df_data['gloss'].append(gloss)
            for key, val in instance.items():
                df_data[key].append(val)
            
        
    lookup_df = pd.DataFrame(df_data)

    # Removing missing rows
    lookup_df = lookup_df[ lookup_df['video_id'].apply(lambda file_name: os.path.isfile(f"dataset/WLASL/videos/{file_name}.mp4"))]
            
    # Getting important columns
    lookup_df = pd.DataFrame(
        data={
            'gloss': lookup_df['gloss'].values,
            'split': lookup_df['split'].values,
            'video_id': lookup_df['video_id'].values
            })

    # Saving lookup
    lookup_df.to_csv("dataset/wlasl_lookup.csv", header=True, index_label=False)