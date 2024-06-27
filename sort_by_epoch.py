import pandas as pd
import re

path = "/Users/sebnemcam/Desktop/Helmholtz/microglia UNet_instance_scores.csv"
df = pd.read_csv(path, delimiter=';',low_memory=False)

print(df.head())

def extract_epoch(patch):
    match = re.search(r'epoch(\d+)', patch)
    return int(match.group(1)) if match else None

df['Epoch'] = df['Patch'].apply(extract_epoch)
df_sorted = df.sort_values(by='Epoch')
df_sorted.to_csv('/Users/sebnemcam/Desktop/Helmholtz/sorted_instance_scores.csv', index=False)

print(df_sorted.head())