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
checkpoint_path="/home/fs01/wc648/protenix/af3-dev/release_model/model_v1.pt"

python3 ./runner/finetune.py \
--run_name protenix_finetune_classifier \
--seed 42 \
--base_dir ./output \
--dtype bf16 \
--project protenix \
--use_deepspeed_evo_attention true \
--use_wandb true \
--diffusion_batch_size 48 \
--eval_interval 5 \
--log_interval 50 \
--checkpoint_interval 400 \
--ema_decay 0.999 \
--train_crop_size 384 \
--max_steps 100000 \
--warmup_steps 2000 \
--lr 0.001 \
--sample_diffusion.N_step 20 \
--load_checkpoint_path ${checkpoint_path} \
--load_ema_checkpoint_path ${checkpoint_path} \
--data.train_sets classifier_table \
--data.test_sets classifier_table_test