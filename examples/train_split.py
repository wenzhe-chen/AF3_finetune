import pandas as pd
from sklearn.model_selection import train_test_split

# Load your CSV file
df = pd.read_csv('2K_screen.csv')

# Split into train and eval (e.g., 80% train, 20% eval)
train_df, eval_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)

# Save to new CSV files
train_df.to_csv('train_2K_screen.csv', index=False)
eval_df.to_csv('eval_2K_screen.csv', index=False)