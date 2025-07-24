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

from typing import Optional, Union

import torch
import torch.nn as nn

from protenix.model.modules.pairformer import PairformerStack
from protenix.model.modules.primitives import LinearNoBias
from protenix.model.modules.deepsettransformer import DeepSetTransformerPooling
from protenix.model.utils import broadcast_token_to_atom, one_hot
from protenix.openfold_local.model.primitives import LayerNorm
from protenix.utils.torch_utils import cdist


class ConfidenceHead(nn.Module):
    """
    Implements Algorithm 31 in AF3
    """

    def __init__(
        self,
        n_blocks: int = 4,
        c_s: int = 384,
        c_z: int = 128,
        c_s_inputs: int = 449,
        b_pae: int = 64,
        b_pde: int = 64,
        b_plddt: int = 50,
        b_resolved: int = 2,
        max_atoms_per_token: int = 20,
        pairformer_dropout: float = 0.0,
        blocks_per_ckpt: Optional[int] = None,
        distance_bin_start: float = 3.375,
        distance_bin_end: float = 21.375,
        distance_bin_step: float = 1.25,
        stop_gradient: bool = True,
        classifier_head: bool = False,
        hidden_channels: int = 64,
    ) -> None:
        """
        Args:
            n_blocks (int, optional): number of blocks for ConfidenceHead. Defaults to 4.
            c_s (int, optional):  hidden dim [for single embedding]. Defaults to 384.
            c_z (int, optional): hidden dim [for pair embedding]. Defaults to 128.
            c_s_inputs (int, optional): hidden dim [for single embedding from InputFeatureEmbedder]. Defaults to 449.
            b_pae (int, optional): the bin number for pae. Defaults to 64.
            b_pde (int, optional): the bin numer for pde. Defaults to 64.
            b_plddt (int, optional): the bin number for plddt. Defaults to 50.
            b_resolved (int, optional): the bin number for resolved. Defaults to 2.
            max_atoms_per_token (int, optional): max atoms in a token. Defaults to 20.
            pairformer_dropout (float, optional): dropout ratio for Pairformer. Defaults to 0.0.
            blocks_per_ckpt: number of Pairformer blocks in each activation checkpoint
            distance_bin_start (float, optional): Start of the distance bin range. Defaults to 3.375.
            distance_bin_end (float, optional): End of the distance bin range. Defaults to 21.375.
            distance_bin_step (float, optional): Step size for the distance bins. Defaults to 1.25.
            stop_gradient (bool, optional): Whether to stop gradient propagation. Defaults to True.
        """
        super(ConfidenceHead, self).__init__()
        self.n_blocks = n_blocks
        self.c_s = c_s
        self.c_z = c_z
        self.c_s_inputs = c_s_inputs
        self.b_pae = b_pae
        self.b_pde = b_pde
        self.b_plddt = b_plddt
        self.b_resolved = b_resolved
        self.max_atoms_per_token = max_atoms_per_token
        self.stop_gradient = stop_gradient
        self.classifier_head = classifier_head
        self.linear_no_bias_s1 = LinearNoBias(
            in_features=self.c_s_inputs, out_features=self.c_z
        )
        self.linear_no_bias_s2 = LinearNoBias(
            in_features=self.c_s_inputs, out_features=self.c_z
        )
        self.bins = nn.Parameter(
            torch.arange(
                start=distance_bin_start, end=distance_bin_end, step=distance_bin_step
            ),
            requires_grad=False,
        )

        self.linear_no_bias_d = LinearNoBias(
            in_features=self.bins.size(dim=0), out_features=self.c_z
        )
        self.pairformer_stack = PairformerStack(
            c_z=self.c_z,
            c_s=self.c_s,
            n_blocks=n_blocks,
            dropout=pairformer_dropout,
            blocks_per_ckpt=blocks_per_ckpt,
        )
        self.linear_no_bias_pae = LinearNoBias(
            in_features=self.c_z, out_features=self.b_pae
        )
        self.linear_no_bias_pde = LinearNoBias(
            in_features=self.c_z, out_features=self.b_pde
        )
        self.plddt_weight = nn.Parameter(
            data=torch.empty(size=(self.max_atoms_per_token, self.c_s, self.b_plddt))
        )
        self.resolved_weight = nn.Parameter(
            data=torch.empty(size=(self.max_atoms_per_token, self.c_s, self.b_resolved))
        )

        self.linear_no_bias_s_inputs = LinearNoBias(self.c_s_inputs, self.c_s)
        self.linear_no_bias_s_trunk = LinearNoBias(self.c_s, self.c_s)
        self.layernorm_s_trunk = LayerNorm(self.c_s)
        self.linear_no_bias_z_trunk = LinearNoBias(self.c_z, self.c_z)
        self.layernorm_z_trunk = LayerNorm(self.c_z)

        # DeepSet Transformer for classification
        if self.classifier_head:
            self.hidden_channels = hidden_channels
            # For z_pair pooling (pairwise features)
            self.z_pair_pooling = DeepSetTransformerPooling(
                n_in=self.c_z, 
                n_hidden_channels=self.hidden_channels, 
                num_heads=8
            )
            # For atom-level features (a)
            self.atom_pooling = DeepSetTransformerPooling(
                n_in=self.c_s, 
                n_hidden_channels=self.hidden_channels, 
                num_heads=8
            )
            # Final classification layer
            self.classifier = nn.Linear(2*self.hidden_channels, 1)  # 2*64 inputs: z_pair_score + atom_score

        with torch.no_grad():
            # Zero init for output layer (before softmax) to zero
            nn.init.zeros_(self.linear_no_bias_pae.weight)
            nn.init.zeros_(self.linear_no_bias_pde.weight)
            nn.init.zeros_(self.plddt_weight)
            nn.init.zeros_(self.resolved_weight)

            # Zero init for trunk embedding input layer
            nn.init.zeros_(self.linear_no_bias_s_trunk.weight)
            nn.init.zeros_(self.linear_no_bias_z_trunk.weight)

    def forward(
        self,
        input_feature_dict: dict[str, Union[torch.Tensor, int, float, dict]],
        s_inputs: torch.Tensor,
        s_trunk: torch.Tensor,
        z_trunk: torch.Tensor,
        pair_mask: torch.Tensor,
        x_pred_coords: torch.Tensor,
        use_memory_efficient_kernel: bool = False,
        use_deepspeed_evo_attention: bool = False,
        use_lma: bool = False,
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            input_feature_dict: Dictionary containing input features.
            s_inputs (torch.Tensor): single embedding from InputFeatureEmbedder
                [..., N_tokens, c_s_inputs]
            s_trunk (torch.Tensor): single feature embedding from PairFormer (Alg17)
                [..., N_tokens, c_s]
            z_trunk (torch.Tensor): pair feature embedding from PairFormer (Alg17)
                [..., N_tokens, N_tokens, c_z]
            pair_mask (torch.Tensor): pair mask
                [..., N_token, N_token]
            x_pred_coords (torch.Tensor): predicted coordinates
                [..., N_sample, N_atoms, 3]
            use_memory_efficient_kernel (bool, optional): Whether to use memory-efficient kernel. Defaults to False.
            use_deepspeed_evo_attention (bool, optional): Whether to use DeepSpeed evolutionary attention. Defaults to False.
            use_lma (bool, optional): Whether to use low-memory attention. Defaults to False.
            inplace_safe (bool, optional): Whether to use inplace operations. Defaults to False.
            chunk_size (Optional[int], optional): Chunk size for memory-efficient operations. Defaults to None.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - a: Atom-level features [N_sample, N_atom, c_s]
                - z_pair: Pairwise features [N_sample, N_tokens, N_tokens, c_z]
        """

        if self.stop_gradient:
            s_inputs = s_inputs.detach()
            s_trunk = s_trunk.detach()
            z_trunk = z_trunk.detach()

        x_rep_atom_mask = input_feature_dict[
            "distogram_rep_atom_mask"
        ].bool()  # [N_atom]
        x_pred_rep_coords = x_pred_coords[..., x_rep_atom_mask, :]
        N_sample = x_pred_rep_coords.size(-3)

        z_init = (
            self.linear_no_bias_s1(s_inputs)[..., None, :, :]
            + self.linear_no_bias_s2(s_inputs)[..., None, :]
        )
        s_init = self.linear_no_bias_s_inputs(s_inputs)
        s_trunk = s_init + self.linear_no_bias_s_trunk(self.layernorm_s_trunk(s_trunk))
        z_trunk = z_init + self.linear_no_bias_z_trunk(self.layernorm_z_trunk(z_trunk))
        if not self.training:
            del z_init
            torch.cuda.empty_cache()

        a_list, z_pair_list = [], []
        for i in range(N_sample):
            a, z_pair = self.memory_efficient_forward(
                input_feature_dict=input_feature_dict,
                s_trunk=s_trunk,
                z_pair=z_trunk,
                pair_mask=pair_mask,
                x_pred_rep_coords=x_pred_rep_coords[..., i, :, :],
                use_memory_efficient_kernel=use_memory_efficient_kernel,
                use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                use_lma=use_lma,
                inplace_safe=inplace_safe,
                chunk_size=chunk_size,
            )
            a_list.append(a)
            z_pair_list.append(z_pair)
        print('a_list',a_list[0].shape)
        print('z_pair_list',z_pair_list[0].shape)
        a = torch.stack(a_list, dim=0)  # [N_sample, N_atom, c_s]
        z_pair = torch.stack(z_pair_list, dim=0)  # [N_sample, N_tokens, N_tokens, c_z]
        return a, z_pair

    def memory_efficient_forward(
        self,
        input_feature_dict: dict[str, Union[torch.Tensor, int, float, dict]],
        s_trunk: torch.Tensor,
        z_pair: torch.Tensor,
        pair_mask: torch.Tensor,
        x_pred_rep_coords: torch.Tensor,
        use_memory_efficient_kernel: bool = False,
        use_deepspeed_evo_attention: bool = False,
        use_lma: bool = False,
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - a: Atom-level features [N_atom, c_s]
                - z_pair: Pairwise features [N_tokens, N_tokens, c_z]
        """
        # Embed pair distances of representative atoms:
        distance_pred = cdist(
            x_pred_rep_coords, x_pred_rep_coords
        )  # [..., N_tokens, N_tokens]
        z_pair = z_pair + self.linear_no_bias_d(
            one_hot(x=distance_pred, v_bins=self.bins)
        )  # [..., N_tokens, N_tokens, c_z]
        # Line 4
        s_single, z_pair = self.pairformer_stack(
            s_trunk,
            z_pair,
            pair_mask,
            use_memory_efficient_kernel=use_memory_efficient_kernel,
            use_deepspeed_evo_attention=use_deepspeed_evo_attention,
            use_lma=use_lma,
            inplace_safe=inplace_safe,
            chunk_size=chunk_size,
        )

        atom_to_token_idx = input_feature_dict[
            "atom_to_token_idx"
        ]  # in range [0, N_token-1] shape: [N_atom]
        # Broadcast s_single: [N_tokens, c_s] -> [N_atoms, c_s]
        a = broadcast_token_to_atom(
            x_token=s_single, atom_to_token_idx=atom_to_token_idx
        )
        z_pair_flat = z_pair.reshape(-1, self.c_z)

        return a, z_pair_flat
