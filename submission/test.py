import pandas as pd

# Read the answer and prediction files
df_ans = pd.read_csv("Dapan_random.csv")
df_pred = pd.read_csv("kaggle_submission.csv")

# Remove .jpg extension from image_name in predictions
df_pred['image_name'] = df_pred['image_name'].str.replace('.jpg', '', regex=False)

# Merge on image_name
df = pd.merge(df_ans, df_pred, on="image_name", suffixes=('_true', '_pred'))

# Compute accuracy
score = (df['label_true'] == df['label_pred']).mean()
print(f"Score (accuracy): {score:.4f}")