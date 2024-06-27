import pandas as pd
import re

path = ""
df = pd.read_csv(path, delimiter=';',low_memory=False)

print(df.head())

def extract_epoch(patch):
    match = re.search(r'epoch(\d+)', patch)
    return int(match.group(1)) if match else None

df['Epoch'] = df['Patch'].apply(extract_epoch)
df_sorted = df.sort_values(by='Epoch')
df = df_sorted.groupby('Epoch')[['TP','FP','FN']].sum().reset_index()
#Dice = 2 |A∩B| / (|A|+|B|) = 2 TP / (2 TP + FP + FN)
dice_scores = pd.Series((2*df['TP']) / (2*df['TP'] + df['FP']+df['FN']) )
df['Dice'] = dice_scores
df.to_csv('/Users/sebnemcam/Desktop/Helmholtz/instance_scores_per_epoch.csv', index=False)