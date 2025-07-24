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

import time
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn

from protenix.model import sample_confidence
from protenix.model.generator import (
    InferenceNoiseScheduler,
    TrainingNoiseSampler,
    sample_diffusion,
    sample_diffusion_training,
)
from protenix.model.utils import simple_merge_dict_list
from protenix.openfold_local.model.primitives import LayerNorm
from protenix.utils.logger import get_logger
from protenix.utils.permutation.permutation import SymmetricPermutation
from protenix.utils.torch_utils import autocasting_disable_decorator
from protenix.data.featurizer import Featurizer

from .modules.confidence import ConfidenceHead
from .modules.diffusion import DiffusionModule
from .modules.embedders import InputFeatureEmbedder, RelativePositionEncoding
from .modules.head import DistogramHead
from .modules.pairformer import MSAModule, PairformerStack, TemplateEmbedder
from .modules.primitives import LinearNoBias
from .modules.classifier import ConfidenceClassifier

logger = get_logger(__name__)


class Protenix(nn.Module):
    """
    Implements Algorithm 1 [Main Inference/Train Loop] in AF3
    """

    def __init__(self, configs) -> None:

        super(Protenix, self).__init__()
        self.configs = configs

        # Some constants
        self.N_cycle = self.configs.model.N_cycle
        self.N_model_seed = self.configs.model.N_model_seed
        self.train_confidence_only = configs.train_confidence_only
        if self.train_confidence_only:  # the final finetune stage
            assert configs.loss.weight.alpha_diffusion == 0.0
            assert configs.loss.weight.alpha_distogram == 0.0

        # Diffusion scheduler
        self.train_noise_sampler = TrainingNoiseSampler(**configs.train_noise_sampler)
        self.inference_noise_scheduler = InferenceNoiseScheduler(
            **configs.inference_noise_scheduler
        )
        self.diffusion_batch_size = self.configs.diffusion_batch_size

        # Model
        self.input_embedder = InputFeatureEmbedder(**configs.model.input_embedder)
        self.relative_position_encoding = RelativePositionEncoding(
            **configs.model.relative_position_encoding
        )
        self.template_embedder = TemplateEmbedder(**configs.model.template_embedder)
        self.msa_module = MSAModule(
            **configs.model.msa_module,
            msa_configs=configs.data.get("msa", {}),
        )
        self.pairformer_stack = PairformerStack(**configs.model.pairformer)
        self.diffusion_module = DiffusionModule(**configs.model.diffusion_module)
        self.distogram_head = DistogramHead(**configs.model.distogram_head)
        self.confidence_head = ConfidenceHead(**configs.model.confidence_head)
        #print("Configs:", self.configs)
        if self.configs['classifier']:
            self.confidence_classifier = ConfidenceClassifier(**configs.model.confidence_classifier)

        self.c_s, self.c_z, self.c_s_inputs = (
            configs.c_s,
            configs.c_z,
            configs.c_s_inputs,
        )
        self.linear_no_bias_sinit = LinearNoBias(
            in_features=self.c_s_inputs, out_features=self.c_s
        )
        self.linear_no_bias_zinit1 = LinearNoBias(
            in_features=self.c_s, out_features=self.c_z
        )
        self.linear_no_bias_zinit2 = LinearNoBias(
            in_features=self.c_s, out_features=self.c_z
        )
        self.linear_no_bias_token_bond = LinearNoBias(
            in_features=1, out_features=self.c_z
        )
        self.linear_no_bias_z_cycle = LinearNoBias(
            in_features=self.c_z, out_features=self.c_z
        )
        self.linear_no_bias_s = LinearNoBias(
            in_features=self.c_s, out_features=self.c_s
        )
        self.layernorm_z_cycle = LayerNorm(self.c_z)
        self.layernorm_s = LayerNorm(self.c_s)

        # Zero init the recycling layer
        nn.init.zeros_(self.linear_no_bias_z_cycle.weight)
        nn.init.zeros_(self.linear_no_bias_s.weight)

    def get_classifier_input(self, summary_confidence, full_data, ligand_length):
        summary_confidence_keys = ['gpde', 'ranking_score', 'chain_ptm', 'chain_iptm', 'chain_plddt','has_clash','disorder']
        full_data_keys = ['token_pair_pae', 'token_pair_pde', 'contact_probs']

        #print('token_pair_pae',full_data[0]['token_pair_pae'].shape)
        #print('token_pair_pde',full_data[0]['token_pair_pde'].shape)
        #print('contact_probs',full_data[0]['contact_probs'].shape)

        if self.configs.model.confidence_classifier.use_intersted_atom_mask:
            summary_confidence_keys.extend(['pb_ranking_score'])

        features = []
        for sample_conf, sample_full in zip(summary_confidence, full_data):
            # Flatten and concatenate all specified keys from summary_confidence
            summary_feats = [
                (sample_conf[k].flatten() / 100 if k == 'chain_plddt' else sample_conf[k].flatten()) if torch.is_tensor(sample_conf[k])
                else (torch.tensor(sample_conf[k]).flatten() / 100 if k == 'chain_plddt' else torch.tensor(sample_conf[k]).flatten())
                for k in summary_confidence_keys
            ]
            # Flatten and concatenate all specified keys from full_data
            full_feats = [
                torch.cat([
                    sample_full[k][-ligand_length:, :].flatten(),
                    sample_full[k][:-ligand_length, -ligand_length:].flatten()
                ], dim=0) if torch.is_tensor(sample_full[k])
                else torch.cat([
                    torch.tensor(sample_full[k][-ligand_length:, :]).flatten(),
                    torch.tensor(sample_full[k][:-ligand_length, -ligand_length:]).T.flatten()
                ], dim=0)
                for k in full_data_keys
            ]
            # Concatenate all features for this sample
            features.append(torch.cat(summary_feats + full_feats, dim=-1))

        # Stack all sample feature vectors into a single tensor with shape [N_sample, feat_dim]
        confidence_scores = torch.stack(features, dim=0)
        return confidence_scores

    def get_pairformer_output(
        self,
        input_feature_dict: dict[str, Any],
        N_cycle: int,
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
    ) -> tuple[torch.Tensor, ...]:
        """
        The forward pass from the input to pairformer output

        Args:
            input_feature_dict (dict[str, Any]): input features
            N_cycle (int): number of cycles
            inplace_safe (bool): Whether it is safe to use inplace operations. Defaults to False.
            chunk_size (Optional[int]): Chunk size for memory-efficient operations. Defaults to None.

        Returns:
            Tuple[torch.Tensor, ...]: s_inputs, s, z
        """
        N_token = input_feature_dict["residue_index"].shape[-1]
        if N_token <= 16:
            # Deepspeed_evo_attention do not support token <= 16
            deepspeed_evo_attention_condition_satisfy = False
        else:
            deepspeed_evo_attention_condition_satisfy = True

        if self.train_confidence_only:
            self.input_embedder.eval()
            self.template_embedder.eval()
            self.msa_module.eval()
            self.pairformer_stack.eval()

        # Line 1-5
        s_inputs = self.input_embedder(
            input_feature_dict, inplace_safe=False, chunk_size=chunk_size
        )  # [..., N_token, 449]
        s_init = self.linear_no_bias_sinit(s_inputs)  #  [..., N_token, c_s]
        z_init = (
            self.linear_no_bias_zinit1(s_init)[..., None, :]
            + self.linear_no_bias_zinit2(s_init)[..., None, :, :]
        )  #  [..., N_token, N_token, c_z]
        if inplace_safe:
            z_init += self.relative_position_encoding(input_feature_dict)
            z_init += self.linear_no_bias_token_bond(
                input_feature_dict["token_bonds"].unsqueeze(dim=-1)
            )
        else:
            z_init = z_init + self.relative_position_encoding(input_feature_dict)
            z_init = z_init + self.linear_no_bias_token_bond(
                input_feature_dict["token_bonds"].unsqueeze(dim=-1)
            )
        # Line 6
        z = torch.zeros_like(z_init)
        s = torch.zeros_like(s_init)

        # Line 7-13 recycling
        for cycle_no in range(N_cycle):
            with torch.set_grad_enabled(
                self.training
                and (not self.train_confidence_only)
                and cycle_no == (N_cycle - 1)
            ):
                z = z_init + self.linear_no_bias_z_cycle(self.layernorm_z_cycle(z))
                if inplace_safe:
                    if self.template_embedder.n_blocks > 0:
                        z += self.template_embedder(
                            input_feature_dict,
                            z,
                            use_memory_efficient_kernel=self.configs.use_memory_efficient_kernel,
                            use_deepspeed_evo_attention=self.configs.use_deepspeed_evo_attention
                            and deepspeed_evo_attention_condition_satisfy,
                            use_lma=self.configs.use_lma,
                            inplace_safe=inplace_safe,
                            chunk_size=chunk_size,
                        )
                    z = self.msa_module(
                        input_feature_dict,
                        z,
                        s_inputs,
                        pair_mask=None,
                        use_memory_efficient_kernel=self.configs.use_memory_efficient_kernel,
                        use_deepspeed_evo_attention=self.configs.use_deepspeed_evo_attention
                        and deepspeed_evo_attention_condition_satisfy,
                        use_lma=self.configs.use_lma,
                        inplace_safe=inplace_safe,
                        chunk_size=chunk_size,
                    )
                else:
                    if self.template_embedder.n_blocks > 0:
                        z = z + self.template_embedder(
                            input_feature_dict,
                            z,
                            use_memory_efficient_kernel=self.configs.use_memory_efficient_kernel,
                            use_deepspeed_evo_attention=self.configs.use_deepspeed_evo_attention
                            and deepspeed_evo_attention_condition_satisfy,
                            use_lma=self.configs.use_lma,
                            inplace_safe=inplace_safe,
                            chunk_size=chunk_size,
                        )
                    z = self.msa_module(
                        input_feature_dict,
                        z,
                        s_inputs,
                        pair_mask=None,
                        use_memory_efficient_kernel=self.configs.use_memory_efficient_kernel,
                        use_deepspeed_evo_attention=self.configs.use_deepspeed_evo_attention
                        and deepspeed_evo_attention_condition_satisfy,
                        use_lma=self.configs.use_lma,
                        inplace_safe=inplace_safe,
                        chunk_size=chunk_size,
                    )
                s = s_init + self.linear_no_bias_s(self.layernorm_s(s))
                s, z = self.pairformer_stack(
                    s,
                    z,
                    pair_mask=None,
                    use_memory_efficient_kernel=self.configs.use_memory_efficient_kernel,
                    use_deepspeed_evo_attention=self.configs.use_deepspeed_evo_attention
                    and deepspeed_evo_attention_condition_satisfy,
                    use_lma=self.configs.use_lma,
                    inplace_safe=inplace_safe,
                    chunk_size=chunk_size,
                )

        if self.train_confidence_only:
            self.input_embedder.train()
            self.template_embedder.train()
            self.msa_module.train()
            self.pairformer_stack.train()

        return s_inputs, s, z

    def sample_diffusion(self, **kwargs) -> torch.Tensor:
        """
        Samples diffusion process based on the provided configurations.

        Returns:
            torch.Tensor: The result of the diffusion sampling process.
        """
        _configs = {
            key: self.configs.sample_diffusion.get(key)
            for key in [
                "gamma0",
                "gamma_min",
                "noise_scale_lambda",
                "step_scale_eta",
            ]
        }
        _configs.update(
            {
                "attn_chunk_size": (
                    self.configs.infer_setting.chunk_size if not self.training else None
                ),
                "diffusion_chunk_size": (
                    self.configs.infer_setting.sample_diffusion_chunk_size
                    if not self.training
                    else None
                ),
            }
        )
        return autocasting_disable_decorator(self.configs.skip_amp.sample_diffusion)(
            sample_diffusion
        )(**_configs, **kwargs)

    def run_confidence_head(self, *args, **kwargs):
        """
        Runs the confidence head with optional automatic mixed precision (AMP) disabled.

        Returns:
            Any: The output of the confidence head.
        """
        return autocasting_disable_decorator(self.configs.skip_amp.confidence_head)(
            self.confidence_head
        )(*args, **kwargs)

    def run_confidence_classifier(self, *args, **kwargs):
        return autocasting_disable_decorator(self.configs.skip_amp.confidence_classifier)(
            self.confidence_classifier
        )(*args, **kwargs) 

    def main_inference_loop(
        self,
        input_feature_dict: dict[str, Any],
        label_dict: dict[str, Any],
        N_cycle: int,
        mode: str,
        inplace_safe: bool = True,
        chunk_size: Optional[int] = 4,
        N_model_seed: int = 1,
        symmetric_permutation: SymmetricPermutation = None,
        atom_array: Any = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
        """
        Main inference loop (multiple model seeds) for the Alphafold3 model.

        Args:
            input_feature_dict (dict[str, Any]): Input features dictionary.
            label_dict (dict[str, Any]): Label dictionary.
            N_cycle (int): Number of cycles.
            mode (str): Mode of operation (e.g., 'inference').
            inplace_safe (bool): Whether to use inplace operations safely. Defaults to True.
            chunk_size (Optional[int]): Chunk size for memory-efficient operations. Defaults to 4.
            N_model_seed (int): Number of model seeds. Defaults to 1.
            symmetric_permutation (SymmetricPermutation): Symmetric permutation object. Defaults to None.

        Returns:
            tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]: Prediction, log, and time dictionaries.
        """
        if self.configs.model.confidence_classifier.use_intersted_atom_mask:
            assert atom_array is not None, "atom_array is required when use_intersted_atom_mask is True"

        pred_dicts = []
        log_dicts = []
        time_trackers = []
        for _ in range(N_model_seed):
            pred_dict, log_dict, time_tracker = self._main_inference_loop(
                input_feature_dict=input_feature_dict,
                label_dict=label_dict,
                N_cycle=N_cycle,
                mode=mode,
                inplace_safe=inplace_safe,
                chunk_size=chunk_size,
                symmetric_permutation=symmetric_permutation,
                atom_array=atom_array,
            )
            pred_dicts.append(pred_dict)
            log_dicts.append(log_dict)
            time_trackers.append(time_tracker)
        #print(f"Keys in pred_dicts: {[list(x.keys()) for x in pred_dicts]}")

        # Combine outputs of multiple models
        def _cat(dict_list, key):
            return torch.cat([x[key] for x in dict_list], dim=0)

        def _list_join(dict_list, key):
            return sum([x[key] for x in dict_list], [])

        all_pred_dict = {
            "coordinate": _cat(pred_dicts, "coordinate"),
            "summary_confidence": _list_join(pred_dicts, "summary_confidence"),
            "full_data": _list_join(pred_dicts, "full_data"),
            "plddt": _cat(pred_dicts, "plddt"),
            "pae": _cat(pred_dicts, "pae"),
            "pde": _cat(pred_dicts, "pde"),
            "resolved": _cat(pred_dicts, "resolved")
        }

        if self.configs['classifier']:
            all_pred_dict['binder']=_cat(pred_dicts, "binder")
            if self.configs.train_classifier_by_inference:
                all_pred_dict['classifier_feats'] = _cat(pred_dicts, "classifier_feats")
        
        if self.configs.train_classifier_by_inference:
            all_pred_dict['confidence_features_a'] = _cat(pred_dicts, "confidence_features_a")
            all_pred_dict['confidence_features_z_pair'] = _cat(pred_dicts, "confidence_features_z_pair")

        #print('all_pred_dict',all_pred_dict)
        all_log_dict = simple_merge_dict_list(log_dicts)
        all_time_dict = simple_merge_dict_list(time_trackers)
        return all_pred_dict, all_log_dict, all_time_dict

    def _main_inference_loop(
        self,
        input_feature_dict: dict[str, Any],
        label_dict: dict[str, Any],
        N_cycle: int,
        mode: str,
        inplace_safe: bool = True,
        chunk_size: Optional[int] = 4,
        symmetric_permutation: SymmetricPermutation = None,
        atom_array: Any = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
        """
        Main inference loop (single model seed) for the Alphafold3 model.

        Returns:
            tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]: Prediction, log, and time dictionaries.
        """
        step_st = time.time()
        N_token = input_feature_dict["residue_index"].shape[-1]
        if N_token <= 16:
            deepspeed_evo_attention_condition_satisfy = False
        else:
            deepspeed_evo_attention_condition_satisfy = True

        log_dict = {}
        pred_dict = {}
        time_tracker = {}

        s_inputs, s, z = self.get_pairformer_output(
            input_feature_dict=input_feature_dict,
            N_cycle=N_cycle,
            inplace_safe=inplace_safe,
            chunk_size=chunk_size,
        )
        if mode == "inference":
            keys_to_delete = []
            for key in input_feature_dict.keys():
                if "template_" in key or key in [
                    "msa",
                    "has_deletion",
                    "deletion_value",
                    "profile",
                    "deletion_mean",
                    "token_bonds",
                ]:
                    keys_to_delete.append(key)

            for key in keys_to_delete:
                del input_feature_dict[key]
            torch.cuda.empty_cache()

        step_trunk = time.time()
        time_tracker.update({"pairformer": step_trunk - step_st})
        # Sample diffusion
        # [..., N_sample, N_atom, 3]
        N_sample = self.configs.sample_diffusion["N_sample"]
        N_step = self.configs.sample_diffusion["N_step"]

        noise_schedule = self.inference_noise_scheduler(
            N_step=N_step, device=s_inputs.device, dtype=s_inputs.dtype
        )
        pred_dict["coordinate"] = self.sample_diffusion(
            denoise_net=self.diffusion_module,
            input_feature_dict=input_feature_dict,
            s_inputs=s_inputs,
            s_trunk=s,
            z_trunk=z,
            N_sample=N_sample,
            noise_schedule=noise_schedule,
            inplace_safe=inplace_safe,
        )

        step_diffusion = time.time()
        time_tracker.update({"diffusion": step_diffusion - step_trunk})
        if mode == "simple_inference" and N_token > 2000:
            torch.cuda.empty_cache()

        # Distogram logits: log contact_probs only, to reduce the dimension
        pred_dict["contact_probs"] = sample_confidence.compute_contact_prob(
            distogram_logits=self.distogram_head(z),
            **sample_confidence.get_bin_params(self.configs.loss.distogram),
        )  # [N_token, N_token]

        # Confidence logits
        (
            pred_dict["plddt"],
            pred_dict["pae"],
            pred_dict["pde"],
            pred_dict["resolved"],
        ) = self.run_confidence_head(
            input_feature_dict=input_feature_dict,
            s_inputs=s_inputs,
            s_trunk=s,
            z_trunk=z,
            pair_mask=None,
            x_pred_coords=pred_dict["coordinate"],
            use_memory_efficient_kernel=self.configs.use_memory_efficient_kernel,
            use_deepspeed_evo_attention=self.configs.use_deepspeed_evo_attention
            and deepspeed_evo_attention_condition_satisfy,
            use_lma=self.configs.use_lma,
            inplace_safe=inplace_safe,
            chunk_size=chunk_size,
        )

        # Run confidence_features.py ConfidenceHead if train_classifier_by_inference is True
        if getattr(self.configs, 'train_classifier_by_inference', False):
            from protenix.model.modules import confidence_features
            confidence_features = confidence_features.ConfidenceHead(**self.configs.model.confidence_head).to(s_inputs.device)
            a, z_pair = confidence_features(
                input_feature_dict=input_feature_dict,
                s_inputs=s_inputs,
                s_trunk=s,
                z_trunk=z,
                pair_mask=None,
                x_pred_coords=pred_dict["coordinate"],
                use_memory_efficient_kernel=self.configs.use_memory_efficient_kernel,
                use_deepspeed_evo_attention=self.configs.use_deepspeed_evo_attention and deepspeed_evo_attention_condition_satisfy,
                use_lma=self.configs.use_lma,
                inplace_safe=inplace_safe,
                chunk_size=chunk_size,
            )
            pred_dict['confidence_features_a'] = a
            pred_dict['confidence_features_z_pair'] = z_pair

        step_confidence = time.time()
        time_tracker.update({"confidence": step_confidence - step_diffusion})
        time_tracker.update({"model_forward": time.time() - step_st})

        if not self.configs['classifier']:
            # Permutation: when label is given, permute coordinates and other heads
            if label_dict is not None and symmetric_permutation is not None:
                pred_dict, log_dict = symmetric_permutation.permute_inference_pred_dict(
                    input_feature_dict=input_feature_dict,
                    pred_dict=pred_dict,
                    label_dict=label_dict,
                    permute_by_pocket=("pocket_mask" in label_dict)
                    and ("interested_ligand_mask" in label_dict),
                )
                last_step_seconds = step_confidence
                time_tracker.update({"permutation": time.time() - last_step_seconds})

        # Summary Confidence & Full Data
        # Computed after coordinates and logits are permuted
        #print (input_feature_dict["asym_id"])
        if not self.configs['model']['confidence_classifier']['use_intersted_atom_mask']:
            interested_atom_mask = None
        else:
            if atom_array is not None:
                coords = pred_dict["coordinate"][0].detach().cpu().numpy().astype(float)
                atom_array.coord = coords
                interested_ligand_mask, pocket_mask = Featurizer.get_lig_pocket_mask(
                    atom_array=atom_array, lig_label_asym_id="B"
                )
                interested_atom_mask = interested_ligand_mask
                print('using interested_atom_mask',interested_atom_mask)
            else:
                print('atom_array is Missing')
                interested_atom_mask = None

        pred_dict["summary_confidence"], pred_dict["full_data"] = (
            sample_confidence.compute_full_data_and_summary(
                configs=self.configs,
                pae_logits=pred_dict["pae"],
                plddt_logits=pred_dict["plddt"],
                pde_logits=pred_dict["pde"],
                contact_probs=pred_dict.get(
                    "per_sample_contact_probs", pred_dict["contact_probs"]
                ),
                token_asym_id=input_feature_dict["asym_id"],
                token_has_frame=input_feature_dict["has_frame"],
                atom_coordinate=pred_dict["coordinate"],
                atom_to_token_idx=input_feature_dict["atom_to_token_idx"],
                atom_is_polymer=1 - input_feature_dict["is_ligand"],
                N_recycle=N_cycle,
                interested_atom_mask=interested_atom_mask,
                return_full_data=True,
                mol_id=(
                    input_feature_dict["mol_id"] if mode != "simple_inference" else None
                ),
                elements_one_hot=(
                    input_feature_dict["ref_element"]
                    if mode != "simple_inference"
                    else None
                ),
            )
        )

        #print('summary_confidence:',pred_dict["summary_confidence"])
        #    'full_data_shape:',pred_dict["full_data"])

        if self.configs['classifier']:

            confidence_scores = self.get_classifier_input(pred_dict["summary_confidence"], pred_dict["full_data"], self.configs.model.confidence_classifier.ligand_length)
            print('confidence_scores_shape:',confidence_scores.shape)

            if self.configs.train_classifier_by_inference:
                pred_dict['classifier_feats'] = confidence_scores

            # Forward pass on GPU
            confidence_output = self.run_confidence_classifier(confidence_scores)
            print('confidence_output:',torch.softmax(confidence_output, dim=1)[:, 1])

            # Store confidence-based classification output
            pred_dict['binder'] = confidence_output
            #print('confidence_output',confidence_output, 'pred_dict[binder]', pred_dict['binder'])

            # Store confidence-based classification output in summary_confidence for the output.json
            for i in range (len(pred_dict["summary_confidence"])):
                 pred_dict["summary_confidence"][i]['prob']= torch.softmax(confidence_output, dim=1)[i, 1]
                 pred_dict["summary_confidence"][i]['binder']= torch.argmax(pred_dict['binder'][i], dim=-1)
            
            # #print(pred_dict["summary_confidence"][0])

        return pred_dict, log_dict, time_tracker

    def main_train_loop(
        self,
        input_feature_dict: dict[str, Any],
        label_full_dict: dict[str, Any],
        label_dict: dict,
        N_cycle: int,
        symmetric_permutation: SymmetricPermutation,
        atom_array: Any = None,
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
     ) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
        """
        Main training loop for the Alphafold3 model.

        Args:
            input_feature_dict (dict[str, Any]): Input features dictionary.
            label_full_dict (dict[str, Any]): Full label dictionary (uncropped).
            label_dict (dict): Label dictionary (cropped).
            N_cycle (int): Number of cycles.
            symmetric_permutation (SymmetricPermutation): Symmetric permutation object.
            inplace_safe (bool): Whether to use inplace operations safely. Defaults to False.
            chunk_size (Optional[int]): Chunk size for memory-efficient operations. Defaults to None.

        Returns:
            tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
                Prediction, updated label, and log dictionaries.
        """
        N_token = input_feature_dict["residue_index"].shape[-1]
        if N_token <= 16:
            deepspeed_evo_attention_condition_satisfy = False
        else:
            deepspeed_evo_attention_condition_satisfy = True

        s_inputs, s, z = self.get_pairformer_output(
            input_feature_dict=input_feature_dict,
            N_cycle=N_cycle,
            inplace_safe=inplace_safe,
            chunk_size=chunk_size,
        )

        #print('s_inputs.grad_fn',s_inputs.grad_fn)
        #print('s.grad_fn',s.grad_fn)
        #print('z.grad_fn',z.grad_fn)

        log_dict = {}
        pred_dict = {}

        if self.configs['classifier']:
            # Grad is needed for classifier
            # [..., 1, N_atom, 3]
                N_sample_mini_rollout = self.configs.sample_diffusion[
                    "N_sample_mini_rollout"
                ]  # =1
                N_step_mini_rollout = self.configs.sample_diffusion["N_step_mini_rollout"]

                coordinate_mini = self.sample_diffusion(
                    denoise_net=self.diffusion_module,
                    input_feature_dict=input_feature_dict,
                    s_inputs=s_inputs,
                    s_trunk=s,
                    z_trunk=z,
                    N_sample=N_sample_mini_rollout,
                    noise_schedule=self.inference_noise_scheduler(
                        N_step=N_step_mini_rollout,
                        device=s_inputs.device,
                        dtype=s_inputs.dtype,
                    ),
                )
                #coordinate_mini.detach_()
                pred_dict["coordinate_mini"] = coordinate_mini
        else:
            # Mini-rollout: used for confidence and label permutation
            with torch.no_grad():
                # [..., 1, N_atom, 3]
                N_sample_mini_rollout = self.configs.sample_diffusion[
                    "N_sample_mini_rollout"
                ]  # =1
                N_step_mini_rollout = self.configs.sample_diffusion["N_step_mini_rollout"]

                coordinate_mini = self.sample_diffusion(
                    denoise_net=self.diffusion_module,
                    input_feature_dict=input_feature_dict,
                    s_inputs=s_inputs.detach(),
                    s_trunk=s.detach(),
                    z_trunk=z.detach(),
                    N_sample=N_sample_mini_rollout,
                    noise_schedule=self.inference_noise_scheduler(
                        N_step=N_step_mini_rollout,
                        device=s_inputs.device,
                        dtype=s_inputs.dtype,
                    ),
                )
                coordinate_mini.detach_()
                pred_dict["coordinate_mini"] = coordinate_mini

                # Permute ground truth to match mini-rollout prediction
                label_dict, perm_log_dict = (
                    symmetric_permutation.permute_label_to_match_mini_rollout(
                        coordinate_mini,
                        input_feature_dict,
                        label_dict,
                        label_full_dict,
                    )
                )
                log_dict.update(perm_log_dict)
        #print('coordinate_mini.grad_fn',pred_dict["coordinate_mini"].grad_fn)

        # Confidence: use mini-rollout prediction, and detach token embeddings
        plddt_pred, pae_pred, pde_pred, resolved_pred = self.run_confidence_head(
            input_feature_dict=input_feature_dict,
            s_inputs=s_inputs,
            s_trunk=s,
            z_trunk=z,
            pair_mask=None,
            x_pred_coords=coordinate_mini,
            use_memory_efficient_kernel=self.configs.use_memory_efficient_kernel,
            use_deepspeed_evo_attention=self.configs.use_deepspeed_evo_attention
            and deepspeed_evo_attention_condition_satisfy,
            use_lma=self.configs.use_lma,
            inplace_safe=inplace_safe,
            chunk_size=chunk_size,
        )
        pred_dict.update(
            {
                "plddt": plddt_pred,
                "pae": pae_pred,
                "pde": pde_pred,
                "resolved": resolved_pred,
            }
        )

        #print('plddt.shape',pred_dict['plddt'].grad_fn)
        #print('coordinate_mini',coordinate_mini.shape)
        #print('asym_id',input_feature_dict["asym_id"])

        if self.configs['classifier']:
            
            # Combine confidence scores
            #plddt_flattened = pred_dict["plddt"].view(pred_dict["plddt"].size(0), -1)
            #pae_flattened = pred_dict["pae"].view(pred_dict["pae"].size(0), -1)  # Flatten last three axes
            #pde_flattened = pred_dict["pde"].view(pred_dict["pde"].size(0), -1)  # Flatten last three axes
            #contact_probs_flattened = pred_dict["contact_probs"].view(pred_dict["contact_probs"].size(0), -1)  # Flatten last three axes
            #resolved_flattened = pred_dict["resolved"].view(pred_dict["resolved"].size(0), -1)
            #confidence_scores = torch.cat([pae_flattened, pde_flattened,contact_probs_flattened], dim=-1)

            # Modified code to reduce dimensionality:
            # Assuming PAE and PDE are [batch, N_res, N_res], take mean across one dimension
            #pae_mean = pred_dict["pae"].mean(dim=-1)  # [batch, N_res]
            #pde_mean = pred_dict["pde"].mean(dim=-1)  # [batch, N_res]
            # Assuming plddt and resolved are [batch, N_res]
           
            pred_dict["contact_probs"] = sample_confidence.compute_contact_prob(
            distogram_logits=self.distogram_head(z),
            **sample_confidence.get_bin_params(self.configs.loss.distogram),
            )  # [N_token, N_token]
            #print('contact_probs.grad_fn',pred_dict["contact_probs"].grad_fn)

            if label_dict is None or not self.configs['model']['confidence_classifier']['use_intersted_atom_mask']:
                interested_atom_mask = None
            else:
                if atom_array is not None:
                    coords = coordinate_mini[0].detach().cpu().numpy().astype(float)
                    atom_array.coord = coords
                    interested_ligand_mask, pocket_mask = Featurizer.get_lig_pocket_mask(
                        atom_array=atom_array, lig_label_asym_id="B"
                    )
                    interested_atom_mask = interested_ligand_mask
                    print('using interested_atom_mask',interested_atom_mask)
                else:
                    print('atom_array is Missing')
                    interested_atom_mask = None

            summary_confidence, full_data = sample_confidence.compute_full_data_and_summary(
                configs=self.configs,
                pae_logits=pred_dict["pae"],
                plddt_logits=pred_dict["plddt"],
                pde_logits=pred_dict["pde"],
                contact_probs=pred_dict.get(
                    "per_sample_contact_probs", pred_dict["contact_probs"]
                ),
                token_asym_id=input_feature_dict["asym_id"],
                token_has_frame=input_feature_dict["has_frame"],
                atom_coordinate=pred_dict["coordinate_mini"],
                atom_to_token_idx=input_feature_dict["atom_to_token_idx"],
                atom_is_polymer=1 - input_feature_dict["is_ligand"],
                N_recycle=N_cycle,
                interested_atom_mask=interested_atom_mask,
                return_full_data=True,
                mol_id=input_feature_dict["mol_id"],
                elements_one_hot=input_feature_dict["ref_element"]
                )
            #print('summary_confidence.grad_fn',summary_confidence.grad_fn)
            #print('full_data.grad_fn',full_data.grad_fn)
            print('summary_confidence:',summary_confidence)
            #print('full_data:',full_data[0].keys())

            confidence_scores = self.get_classifier_input(summary_confidence, full_data, self.configs.model.confidence_classifier.ligand_length)
            # if torch.argmax(label_dict, dim=-1) == 0:
            #     gen1 = torch.Generator().manual_seed(0)
            #     random_tensor1 = torch.randn(1, 24106, generator=gen1).to(confidence_scores.device)
            #     confidence_scores = random_tensor1
            # else:
            #     gen2 = torch.Generator().manual_seed(1)
            #     random_tensor2 = torch.randn(1, 24106, generator=gen2).to(confidence_scores.device)
            #     confidence_scores = random_tensor2

            print('confidence_scores_shape:',confidence_scores.shape)

            # Forward pass on GPU
            confidence_output = self.run_confidence_classifier(confidence_scores)

            # Store confidence-based classification output
            pred_dict['binder'] = confidence_output
            #print('confidence_output',confidence_output, 'pred_dict[binder]', pred_dict['binder'])
                

        if self.train_confidence_only:
            # Skip diffusion loss and distogram loss. Return now.
            return pred_dict, label_dict, log_dict

        if not self.configs['classifier']:
            # Denoising: use permuted coords to generate noisy samples and perform denoising
            # x_denoised: [..., N_sample, N_atom, 3]
            # x_noise_level: [..., N_sample]
            N_sample = self.diffusion_batch_size
            _, x_denoised, x_noise_level = autocasting_disable_decorator(
                self.configs.skip_amp.sample_diffusion_training
            )(sample_diffusion_training)(
                noise_sampler=self.train_noise_sampler,
                denoise_net=self.diffusion_module,
                label_dict=label_dict,
                input_feature_dict=input_feature_dict,
                s_inputs=s_inputs,
                s_trunk=s,
                z_trunk=z,
                N_sample=N_sample,
                diffusion_chunk_size=self.configs.diffusion_chunk_size,
            )
            pred_dict.update(
                {
                    "distogram": self.distogram_head(z),
                    # [..., N_sample=48, N_atom, 3]: diffusion loss
                    "coordinate": x_denoised,
                    "noise_level": x_noise_level,
                }
            )

            # Permute symmetric atom/chain in each sample to match true structure
            # Note: currently chains cannot be permuted since label is cropped
            pred_dict, perm_log_dict, _, _ = (
                symmetric_permutation.permute_diffusion_sample_to_match_label(
                    input_feature_dict, pred_dict, label_dict, stage="train"
                )
            )
            log_dict.update(perm_log_dict)

        return pred_dict, label_dict, log_dict

    def forward(
        self,
        input_feature_dict: dict[str, Any],
        label_full_dict: dict[str, Any],
        label_dict: dict[str, Any],
        mode: str = "inference",
        atom_array: Any = None,
        current_step: Optional[int] = None,
        symmetric_permutation: SymmetricPermutation = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
        """
        Forward pass of the Alphafold3 model.

        Args:
            input_feature_dict (dict[str, Any]): Input features dictionary.
            label_full_dict (dict[str, Any]): Full label dictionary (uncropped).
            label_dict (dict[str, Any]): Label dictionary (cropped).
            mode (str): Mode of operation ('train', 'inference', 'eval'). Defaults to 'inference'.
            current_step (Optional[int]): Current training step. Defaults to None.
            symmetric_permutation (SymmetricPermutation): Symmetric permutation object. Defaults to None.

        Returns:
            tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
                Prediction, updated label, and log dictionaries.
        """

        assert mode in ["train", "inference", "eval"]
        inplace_safe = not (self.training or torch.is_grad_enabled())
        chunk_size = self.configs.infer_setting.chunk_size if inplace_safe else None

        if mode == "train":
            nc_rng = np.random.RandomState(current_step)
            N_cycle = nc_rng.randint(1, self.N_cycle + 1)
            assert self.training
            assert label_dict is not None
            assert symmetric_permutation is not None
            
            pred_dict, label_dict, log_dict = self.main_train_loop(
                input_feature_dict=input_feature_dict,
                label_full_dict=label_full_dict,
                label_dict=label_dict,
                N_cycle=N_cycle,
                symmetric_permutation=symmetric_permutation,
                inplace_safe=inplace_safe,
                chunk_size=chunk_size, 
                atom_array=atom_array,
            )
        elif mode == "inference":
            pred_dict, log_dict, time_tracker = self.main_inference_loop(
                input_feature_dict=input_feature_dict,
                label_dict=None,
                N_cycle=self.N_cycle,
                mode=mode,
                inplace_safe=inplace_safe,
                chunk_size=chunk_size,
                N_model_seed=self.N_model_seed,
                symmetric_permutation=None,
                atom_array=atom_array,
            )
            log_dict.update({"time": time_tracker})
        elif mode == "eval":
            if label_dict is not None:
                if not self.configs['classifier']:
                    assert (
                        label_dict["coordinate"].size()
                        == label_full_dict["coordinate"].size()
                    )
                    label_dict.update(label_full_dict)

            pred_dict, log_dict, time_tracker = self.main_inference_loop(
                input_feature_dict=input_feature_dict,
                label_dict=label_dict,
                N_cycle=self.N_cycle,
                mode=mode,
                inplace_safe=inplace_safe,
                chunk_size=chunk_size,
                N_model_seed=self.N_model_seed,
                symmetric_permutation=symmetric_permutation,
                atom_array=atom_array,
            )
            log_dict.update({"time": time_tracker})

        return pred_dict, label_dict, log_dict
