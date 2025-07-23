"""Load labeled dataset from log file names."""

import os
import pandas as pd


def load_dataset(dirpath: str):
    """Create dataset from files in directory."""
    
    data = []
    entries = os.listdir(dirpath)
    
    for filename in entries:
        if not os.path.isfile(os.path.join(dirpath, filename)):
            continue
        parts = filename.split(".")[0].split("_")
        data.append({
            "project": int(parts[0]),
            "id": int(parts[1]),
            "brown": int(parts[2]),
            "flaky": int(parts[3]),
            "flaky_reason": int(parts[4]) 
        })
    df = pd.DataFrame(data)
    return df   
        
        
        
if __name__ == "__main__":
    print(load_dataset("data/logs/raw/client/").head(1))
