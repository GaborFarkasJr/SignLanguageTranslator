from my_utils import *
import pandas as pd
import numpy as np
import argparse

# Run this if glossary is not added
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Adding glossary to the .pt model file. Only Run this if missing")
    parser.add_argument("-mp", "--model_path", metavar="<model_path>", help="Enter the path of the .pt model file", required=True)
    parser.add_argument("-lp", "--lookup_path", metavar="<lookup_path>", help="Enter the path of the csv lookup the model was trained on", required=True)
    args = parser.parse_args()
    
    model_path = args.model_path
    lookup_path = args.lookup_path
    
    # Training uses the first 10 words from the LSA64 dataset
    lookup_df = pd.read_csv(lookup_path,converters={'video_id': str})
    glossary = np.unique(lookup_df['gloss'].values)
    glossary = glossary
    
    add_model_labels(model_path, glossary)
    
    print("Updated!")