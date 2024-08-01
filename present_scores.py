import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Dice scores for each fold
fold1 = pd.read_csv('')
fold2 = pd.read_csv('')
fold3 = pd.read_csv('')
fold4 = pd.read_csv('')
fold5 = pd.read_csv('')

# no NaN values
fold1['Dice'] = fold1['Dice'].dropna()
fold2['Dice'] = fold2['Dice'].dropna()
fold3['Dice'] = fold3['Dice'].dropna()
fold4['Dice'] = fold4['Dice'].dropna()
fold5['Dice'] = fold5['Dice'].dropna()

fold_dice_scores = [
    fold1['Dice'],
    fold2['Dice'],
    fold3['Dice'],
    fold4['Dice'],
    fold5['Dice']
]

# Calculate summary statistics
mean_scores = [np.mean(scores) for scores in fold_dice_scores]
std_scores = [np.std(scores) for scores in fold_dice_scores]
min_scores = [np.min(scores) for scores in fold_dice_scores]
max_scores = [np.max(scores) for scores in fold_dice_scores]

# Overall statistics
overall_mean = np.mean(mean_scores)
overall_std = np.mean(std_scores)
overall_min = np.min(min_scores)
overall_max = np.max(max_scores)

# Create a summary table
summary_table = pd.DataFrame({
    'Fold': range(1, 6),
    'Mean Dice Score': mean_scores,
    'Standard Deviation': std_scores,
    'Min Dice Score': min_scores,
    'Max Dice Score': max_scores
})
summary_table.loc['Overall'] = ['Overall', overall_mean, overall_std, overall_min, overall_max]

# Print the summary table
print(summary_table)

summary_table.to_csv('', index=False)

# Box plot for visualization

fold_dice_scores = pd.DataFrame({
    'fold1': fold1['Dice'],
    'fold2': fold2['Dice'],
    'fold3':fold3['Dice'],
    'fold4':fold4['Dice'],
    'fold5':fold5['Dice']}
)

fold_dice_scores.boxplot(grid=True)
plt.title('Instance Dice Scores Across 5 Folds')
plt.xlabel('Fold')
plt.ylabel('Dice Score')
plt.xticks(ticks=range(1, 6), labels=range(1, 6))
plt.show()
plt.savefig('')



