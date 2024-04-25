import pandas as pd
import os
import argparse
import tqdm
import random

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str,
                    default='datasets/P4_CPU100/train', help='dataset dir path')
parser.add_argument('--output', type=str,
                    default='datasets/P4_CPU100/train/merged.csv', help='output csv path')

args = parser.parse_args()
dataset_dir = args.dataset
output_path = args.output
# list all csv files
csv_files = os.listdir(dataset_dir)
# filter csv files
csv_files = [csv_file for csv_file in csv_files if csv_file.endswith('.csv')]
random.shuffle(csv_files)

merged_csv = pd.DataFrame()
csv_dfs = list()
print("Reading all csvs...")
for csv_file in tqdm.tqdm(csv_files):
    csv_basename = os.path.basename(csv_file)
    csv_path = os.path.join(dataset_dir, csv_file)
    csv_data = pd.read_csv(csv_path)
    # 只要类型为float(1)的
    if (csv_data['input_type']!=1).any():
        print(f"Skip {csv_basename} because of input_type")
        continue
    
    if (csv_data['space']<-1000).any():
        print(f"Skip {csv_basename} because of space")
        continue
    
    if 'train' in csv_basename or 'eval' in csv_basename:
        print(f"Skip {csv_basename} because of train, eval")
        continue
    csv_data['filename'] = csv_basename
    csv_dfs.append(csv_data)

# merge all csvs
print("Merging all csvs...")
merged_csv = pd.concat(csv_dfs) 

# save to csv
print(f"Saving to csv file: {output_path}")
merged_csv.to_csv(output_path, index=False)