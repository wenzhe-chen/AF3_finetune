import pandas as pd
import math

# Read the input CSV
input_path = "/gpfs/data/hlin-lab/wenzhechen/applications/protenix/examples/2K_screen.csv"
df = pd.read_csv(input_path)

n_splits = 6
rows = len(df)
chunk_size = math.ceil(rows / n_splits)

for i in range(n_splits):
    start = i * chunk_size
    end = min((i + 1) * chunk_size, rows)
    df_chunk = df.iloc[start:end]
    output_path = f"/gpfs/data/hlin-lab/wenzhechen/applications/protenix/examples/2K_screen_{i}.csv"
    df_chunk.to_csv(output_path, index=False)
    print(f"Wrote {output_path} with {len(df_chunk)} rows.")
