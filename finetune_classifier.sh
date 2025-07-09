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

export LAYERNORM_TYPE=fast_layernorm
#checkpoint_path="/home/fs01/wc648/protenix/af3-dev/release_model/model_v1.pt"
checkpoint_path="/home/fs01/wc648/protenix/output/protenix_finetune_classifier_diffusion_only_DDP_20250605_025434/checkpoints/149_ema_0.999.pt"
#checkpoint_path_classifier="/home/fs01/wc648/protenix/output/confidence_classifier_2K_screen.pt"



python3 ./runner/finetune.py \
--run_name protenix_finetune_classifier_pair-only-test_auc \
--seed 42 \
--base_dir ./output \
--dtype bf16 \
--project protenix \
--use_deepspeed_evo_attention true \
--use_wandb false \
--train_classifier_only false \
--iters_to_accumulate 64 \
--diffusion_batch_size 48 \
--eval_interval 1 \
--log_interval 1 \
--checkpoint_interval 20 \
--ema_decay 0.999 \
--train_crop_size 384 \
--max_steps 100000 \
--warmup_steps 2000 \
--lr 0.001 \
--sample_diffusion.N_step 20 \
--sample_diffusion.N_sample 1 \
--load_checkpoint_path ${checkpoint_path} \
--load_classifier_checkpoint false \
--load_ema_checkpoint_path ${checkpoint_path} \
--data.train_sets classifier_table \
--data.test_sets classifier_table_test \