#!/bin/bash

# Copyright 2024 ByteDance and/or its affiliates.
#
# Licensed under the Attribution-NonCommercial 4.0 International
# License (the "License"); you may not use this file except in
# compliance with the License. You may obtain a copy of the
# License at

#     https://creativecommons.org/licenses/by-nc/4.0/

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Example feature and label paths (replace with your actual .pt files)
feat_paths=(
    "/home/fs01/wc648/protenix/output/feats_2K_screen_0.pt"
    "/home/fs01/wc648/protenix/output/feats_2K_screen_1.pt"
    "/home/fs01/wc648/protenix/output/feats_2K_screen_2.pt"
    "/home/fs01/wc648/protenix/output/feats_2K_screen_3.pt"
    "/home/fs01/wc648/protenix/output/feats_2K_screen_4.pt"
    "/home/fs01/wc648/protenix/output/feats_2K_screen_5.pt"
    # Add more as needed
)
label_paths=(
    "/home/fs01/wc648/protenix/output/labels_2K_screen_0.pt"
    "/home/fs01/wc648/protenix/output/labels_2K_screen_1.pt"
    "/home/fs01/wc648/protenix/output/labels_2K_screen_2.pt"
    "/home/fs01/wc648/protenix/output/labels_2K_screen_3.pt"
    "/home/fs01/wc648/protenix/output/labels_2K_screen_4.pt"
    "/home/fs01/wc648/protenix/output/labels_2K_screen_5.pt"
    # Add more as needed
)

# Convert arrays to space-separated strings
feat_paths_str="${feat_paths[@]}"
label_paths_str="${label_paths[@]}"

ligand_length=8  # Set this to your actual ligand length

python3 runner/train_confidence_classifier.py \
    --feat_path $feat_paths_str \
    --label_path $label_paths_str \
    --ligand_length $ligand_length \
    --batch_size 1024 \
    --epochs 10000 \
    --lr 0.0005 \
    --output ./output/confidence_classifier_2K_screen.pt \
    --number_of_chains 2 \
    --patience 1000 \
    # Add --use_intersted_atom_mask if needed
