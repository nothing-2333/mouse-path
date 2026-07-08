import json
import os
  
if __name__ == "__main__":
    root_dir = "./data/click"
    
    length = 0
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            # 判断后缀为 .json
            if filename.lower().endswith(".json"):
                file_path = os.path.join(dirpath, filename)
                
                
                with open(file_path, 'r') as file:
                    data = json.load(file)
                length += len(data)
                print(f"{file_path}: {len(data)}")
    print("总长度:", length)