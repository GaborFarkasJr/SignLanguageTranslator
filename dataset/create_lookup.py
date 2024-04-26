import pandas as pd

# Converting the two lookups to a combined lookup with some additional information
if __name__ == "__main__":
    
    # Loading the datasets lookups
    train_df = pd.read_csv("LSA64_videos_train.csv", converters={'Video_name': str})
    test_df = pd.read_csv("LSA64_videos_test.csv", converters={'Video_name': str})
    
    # Adding split information
    train_df.insert(1, "split", "train")
    test_df.insert(1, "split", "test")
    
    # Combining dataset
    combined_df = pd.concat([train_df, test_df], ignore_index=True, sort=False)
    
    # Change columns around
    combined_df = combined_df[['class_name', 'split', 'video_name']]
    
    # Changing column names
    combined_df.columns = ['gloss', 'split', 'video_id']
    
    # Clear file type
    combined_df["video_id"] = combined_df["video_id"].map(lambda id: id.replace(r".mp4", ""), na_action=None)
    
    # Remove Nan values
    combined_df = combined_df.dropna()
    
    combined_df.to_csv("lsa64/LSA64_lookup.csv", header=True, index_label=False)