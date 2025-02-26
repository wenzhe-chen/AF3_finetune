#!/bin/bash
#SBATCH --job-name=download_job    # Job name
#SBATCH --output=download.log      # Log file
#SBATCH --partition=eht45_0001            # Partition (adapt to your cluster)
#SBATCH --ntasks=1                 # Number of tasks
#SBATCH --time=3-01:00:00            # Maximum runtime (adjust as needed)

wget -c -P ./af3-dev/release_data/ https://af3-dev.tos-cn-beijing.volces.com/release_data.tar.gz

