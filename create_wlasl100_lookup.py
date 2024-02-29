import json
import pandas as pd
import numpy as np

if __name__ == "__main__":
    '''
    Before running this one, create the lookup of all the videos first.
    '''
    # Loading necessary files
    lookup_df = pd.read_csv("dataset/wlasl_lookup.csv", converters={'video_id': str})
    
    wlasl_100_dict = {}
    with open("dataset/WLASL/nslt_100.json", "r") as file:
        wlasl_100_dict  = json.load(file)
        
    # Making a list of video ids
    video_ids = wlasl_100_dict.keys()
    
    # Filtering
    lookup_100_df = lookup_df.loc[lookup_df['video_id'].isin(video_ids)]
    
    # Saving lookup
    lookup_100_df.to_csv("dataset/wlasl_100_lookup.csv", header=True, index_label=False)
    
    # Checking some details
    glossary = np.unique(lookup_100_df['gloss'].values)
    
    print(lookup_100_df)
    print(f"Glossary Count: {len(glossary)}")
    print(glossary)

        
        