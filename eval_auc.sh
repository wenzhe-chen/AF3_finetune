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

N_sample=1
N_step=200
N_cycle=10
seed=101
use_deepspeed_evo_attention=true
input_json_path="/home/fs01/wc648/protenix/examples/eval_2K_screen.csv"
# wget -P /af3-dev/release_model/ https://af3-dev.tos-cn-beijing.volces.com/release_model/model_v1.pt
#load_checkpoint_path="/home/fs01/wc648/protenix/output/protenix_finetune_classifier_2K_screen_2_20250602_080906/checkpoints/799.pt"
load_checkpoint_path="/home/fs01/wc648/protenix/output/protenix_finetune_classifier_diffusion_only_DDP_20250605_025434/checkpoints/199_ema_0.999.pt"
dump_dir="./output"

python3 runner/eval_auc.py \
--classifier true \
--seeds ${seed} \
--load_checkpoint_path ${load_checkpoint_path} \
--dump_dir ${dump_dir} \
--input_json_path ${input_json_path} \
--use_deepspeed_evo_attention ${use_deepspeed_evo_attention} \
--model.N_cycle ${N_cycle} \
--sample_diffusion.N_sample ${N_sample} \
--sample_diffusion.N_step ${N_step} \
--train_classifier_by_inference true