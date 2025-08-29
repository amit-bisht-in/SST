# In your file: check_data.py

import numpy as np
import os

file_path = 'data/data_2d_h36m_cpn_ft_h36m_dbb.npz'

print(f"--- Exploring the structure of: {file_path} ---")

if not os.path.exists(file_path):
    print("Error: File not found.")
else:
    try:
        data = np.load(file_path, allow_pickle=True)
        print("File loaded successfully. Here's what's inside:\n")

        # Loop through all the data arrays stored in the file
        for key in data.keys():
            item = data[key]
            print(f"--> Key: '{key}'")
            print(f"    Type: {type(item)}")
            
            # If it's a numpy array, print more details
            if isinstance(item, np.ndarray):
                print(f"    Shape: {item.shape}")
                print(f"    Data Type: {item.dtype}")

                # If it's a 0-dim array holding a Python object (like a dict)
                if item.shape == () and item.dtype == 'object':
                    print("    This key contains a Python object. Let's look inside it:")
                    obj = item.item()
                    print(f"    Object Type: {type(obj)}")
                    if isinstance(obj, dict):
                        # Print the keys of the dictionary inside
                        print(f"    Object is a dictionary with keys: {list(obj.keys())}")

            print("-" * 20)

    except Exception as e:
        print(f"\nAn error occurred during exploration: {e}")