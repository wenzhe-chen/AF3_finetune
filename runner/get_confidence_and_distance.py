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

import logging
import copy
import os
import traceback
from contextlib import nullcontext
from os.path import exists as opexists
from os.path import join as opjoin
from typing import Any, Mapping
import numpy as np
import torch
import torch.distributed as dist

from configs.configs_base import configs as configs_base
from configs.configs_data import data_configs
from configs.configs_inference import inference_configs
from protenix.config import parse_configs, parse_sys_args
from protenix.data.infer_data_pipeline import get_inference_dataloader
from protenix.model.protenix import Protenix
from protenix.utils.distributed import DIST_WRAPPER
from protenix.utils.seed import seed_everything
from protenix.utils.torch_utils import to_device
from runner.dumper import DataDumper
from sklearn.metrics import roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class InferenceRunner(object):
    def __init__(self, configs: Any) -> None:
        self.configs = configs
        self.init_env()
        self.init_basics()
        self.init_model()
        self.load_checkpoint()
        self.init_dumper(need_atom_confidence=configs.need_atom_confidence)

    def init_env(self) -> None:
        self.print(
            f"Distributed environment: world size: {DIST_WRAPPER.world_size}, "
            + f"global rank: {DIST_WRAPPER.rank}, local rank: {DIST_WRAPPER.local_rank}"
        )
        self.use_cuda = torch.cuda.device_count() > 0
        if self.use_cuda:
            self.device = torch.device("cuda:{}".format(DIST_WRAPPER.local_rank))
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            all_gpu_ids = ",".join(str(x) for x in range(torch.cuda.device_count()))
            devices = os.getenv("CUDA_VISIBLE_DEVICES", all_gpu_ids)
            logging.info(
                f"LOCAL_RANK: {DIST_WRAPPER.local_rank} - CUDA_VISIBLE_DEVICES: [{devices}]"
            )
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")
        if DIST_WRAPPER.world_size > 1:
            dist.init_process_group(backend="nccl")
        if self.configs.use_deepspeed_evo_attention:
            env = os.getenv("CUTLASS_PATH", None)
            self.print(f"env: {env}")
            assert (
                env is not None
            ), "if use ds4sci, set env as https://www.deepspeed.ai/tutorials/ds4sci_evoformerattention/"
            if env is not None:
                logging.info(
                    "The kernels will be compiled when DS4Sci_EvoformerAttention is called for the first time."
                )
        use_fastlayernorm = os.getenv("LAYERNORM_TYPE", None)
        if use_fastlayernorm == "fast_layernorm":
            logging.info(
                "The kernels will be compiled when fast_layernorm is called for the first time."
            )

        logging.info("Finished init ENV.")

    def init_basics(self) -> None:
        self.dump_dir = self.configs.dump_dir
        self.error_dir = opjoin(self.dump_dir, "ERR")
        os.makedirs(self.dump_dir, exist_ok=True)
        os.makedirs(self.error_dir, exist_ok=True)

    def init_model(self) -> None:
        self.model = Protenix(self.configs).to(self.device)

    def load_checkpoint(self) -> None:
        checkpoint_path = self.configs.load_checkpoint_path
        if not os.path.exists(checkpoint_path):
            raise Exception(f"Given checkpoint path not exist [{checkpoint_path}]")
        self.print(
            f"Loading from {checkpoint_path}, strict: {self.configs.load_strict}"
        )
        checkpoint = torch.load(checkpoint_path, self.device)

        sample_key = [k for k in checkpoint["model"].keys()][0]
        self.print(f"Sampled key: {sample_key}")
        if sample_key.startswith("module."):  # DDP checkpoint has module. prefix
            checkpoint["model"] = {
                k[len("module.") :]: v for k, v in checkpoint["model"].items()
            }
        self.model.load_state_dict(
            state_dict=checkpoint["model"],
            strict=False,
        )
        self.model.eval()
        self.print(f"Finish loading checkpoint.")

    def init_dumper(self, need_atom_confidence: bool = False):
        self.dumper = DataDumper(
            base_dir=self.dump_dir, need_atom_confidence=need_atom_confidence
        )

    # Adapted from runner.train.Trainer.evaluate
    @torch.no_grad()
    def predict(self, data: Mapping[str, Mapping[str, Any]]) -> dict[str, torch.Tensor]:
        eval_precision = {
            "fp32": torch.float32,
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
        }[self.configs.dtype]

        enable_amp = (
            torch.autocast(device_type="cuda", dtype=eval_precision)
            if torch.cuda.is_available()
            else nullcontext()
        )

        data = to_device(data, self.device)
        with enable_amp:
            prediction, _, _ = self.model(
                input_feature_dict=data["input_feature_dict"],
                label_full_dict=None,
                label_dict=None,
                mode="inference",
                atom_array=data["atom_array"],
            )

        return prediction

    def print(self, msg: str):
        if DIST_WRAPPER.rank == 0:
            logger.info(msg)


def get_protein_sequence(atom_array, chain_id):
    """
    Extract protein sequence from AtomArray for a specific chain.
    
    Args:
        atom_array: The AtomArray containing the structure
        chain_id: Chain ID to filter by
    
    Returns:
        str: Protein sequence as one-letter amino acid codes
    """
    # One-letter to three-letter amino acid mapping
    aa_mapping = {
        'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
        'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
        'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
        'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
        'SEC': 'U', 'PYL': 'O'  # Selenocysteine and Pyrrolysine
    }
    
    # Debug: Print available chain IDs
    print(f"Available chain IDs: {np.unique(atom_array.label_asym_id)}")
    print(f"Looking for chain: '{chain_id}'")
    
    # Filter atoms for the specific chain
    chain_mask = atom_array.label_asym_id == chain_id
    chain_atoms = atom_array[chain_mask]
    
    print(f"Found {len(chain_atoms)} atoms in chain {chain_id}")
    
    if len(chain_atoms) == 0:
        print(f"No atoms found for chain {chain_id}")
        return ""
    
    # Get unique residues in order
    residues = []
    for i in range(len(chain_atoms)):
        res_info = (chain_atoms[i].res_id, chain_atoms[i].res_name)
        if res_info not in residues:
            residues.append(res_info)
    
    #print(f"Found {len(residues)} unique residues in chain {chain_id}")
    #print(f"First 10 residues: {residues[:10]}")
    
    # Convert to sequence
    sequence = ""
    for res_id, res_name in residues:
        if res_name in aa_mapping:
            sequence += aa_mapping[res_name]
        else:
            sequence += 'X'  # Unknown amino acid
    
    return sequence

def find_dhhc_motif(atom_array, chain_id):
    """
    Find DHHC motif in the protein sequence and return the cysteine residue info.
    
    Args:
        atom_array: The AtomArray containing the structure
        chain_id: Chain ID to search in
    
    Returns:
        tuple: (res_id, res_name, atom_name) for the cysteine sulfur atom
    """
    sequence = get_protein_sequence(atom_array, chain_id)
    print(f"Chain {chain_id} sequence: {sequence}")
    
    if not sequence:
        print(f"No sequence found for chain {chain_id}")
        return None
    
    # Find DHHC motif
    dhhc_pos = sequence.find('DHHC')
    if dhhc_pos == -1:
        print(f"DHHC motif not found in chain {chain_id}")
        return None
    
    print(f"DHHC motif found at position {dhhc_pos} in sequence")
    
    # Get the cysteine position (4th position in DHHC = position 3)
    cys_pos_in_motif = 3
    cys_pos_in_seq = dhhc_pos + cys_pos_in_motif
    
    # Get residue info for the cysteine
    chain_mask = atom_array.label_asym_id == chain_id
    chain_atoms = atom_array[chain_mask]
    
    # Get unique residues in order
    residues = []
    for i in range(len(chain_atoms)):
        res_info = (chain_atoms[i].res_id, chain_atoms[i].res_name)
        if res_info not in residues:
            residues.append(res_info)
    
    if cys_pos_in_seq < len(residues):
        cys_res_id, cys_res_name = residues[cys_pos_in_seq]
        print(f"Cysteine in DHHC motif: residue {cys_res_id} {cys_res_name}")
        return (cys_res_id, cys_res_name, 'SG')  # SG is the sulfur atom in cysteine
    else:
        print(f"Cysteine position {cys_pos_in_seq} out of range")
        return None

def find_p1l_residue(atom_array, chain_id):
    """
    Find P1L residue in the protein structure.
    
    Args:
        atom_array: The AtomArray containing the structure
        chain_id: Chain ID to search in
    
    Returns:
        tuple: (res_id, res_name, atom_name) for the P1L SG atom
    """
    # Debug: Print available chain IDs
    print(f"Available chain IDs for P1L search: {np.unique(atom_array.label_asym_id)}")
    print(f"Looking for P1L in chain: '{chain_id}'")
    
    # Filter atoms for the specific chain
    chain_mask = atom_array.label_asym_id == chain_id
    chain_atoms = atom_array[chain_mask]
    
    print(f"Found {len(chain_atoms)} atoms in chain {chain_id}")
    
    if len(chain_atoms) == 0:
        print(f"No atoms found for chain {chain_id}")
        return None
    
    # Find P1L residue
    p1l_mask = chain_atoms.res_name == 'P1L'
    print(f"P1L mask sum: {np.sum(p1l_mask)}")
    
    if not np.any(p1l_mask):
        print(f"P1L residue not found in chain {chain_id}")
        # Debug: Show available residue names
        unique_res_names = np.unique(chain_atoms.res_name)
        print(f"Available residue names in chain {chain_id}: {unique_res_names}")
        return None
    
    p1l_atoms = chain_atoms[p1l_mask]
    p1l_res_id = p1l_atoms[0].res_id
    p1l_res_name = p1l_atoms[0].res_name
    
    print(f"P1L residue found: {p1l_res_id} {p1l_res_name}")
    
    # Check if SG atom exists
    sg_mask = (p1l_atoms.atom_name == 'SG')
    print(f"SG atoms in P1L: {np.sum(sg_mask)}")
    
    if not np.any(sg_mask):
        print(f"SG atom not found in P1L residue")
        # Debug: Show available atom names in P1L
        unique_atom_names = np.unique(p1l_atoms.atom_name)
        print(f"Available atom names in P1L: {unique_atom_names}")
        return None
    
    return (p1l_res_id, p1l_res_name, 'SG')

def calculate_atom_distance(atom_array, chain1, res1, res_name1, atom1, chain2, res2, res_name2, atom2):
    """
    Calculate distance between two specific atoms.
    """
    # Find the first atom
    mask1 = (
        (atom_array.label_asym_id == chain1) &
        (atom_array.res_id == res1) &
        (atom_array.res_name == res_name1) &
        (atom_array.atom_name == atom1)
    )
    
    # Find the second atom
    mask2 = (
        (atom_array.label_asym_id == chain2) &
        (atom_array.res_id == res2) &
        (atom_array.res_name == res_name2) &
        (atom_array.atom_name == atom2)
    )
    
    # Get the coordinates
    coord1 = atom_array.coord[mask1]
    coord2 = atom_array.coord[mask2]
    
    if len(coord1) == 0 or len(coord2) == 0:
        raise ValueError(f"Could not find atoms: {chain1} {res1} {res_name1} {atom1} or {chain2} {res2} {res_name2} {atom2}")
    
    # Calculate Euclidean distance
    distance = np.linalg.norm(coord1[0] - coord2[0])
    return distance



def main(configs: Any) -> None:
    # Runner
    runner = InferenceRunner(configs)

    # Data
    logger.info(f"Loading data from\n{configs.input_json_path}")
    dataloader = get_inference_dataloader(configs=configs)
    runname = configs.input_json_path.split('/')[-1].split('.')[0]

    num_data = len(dataloader.dataset)
    for seed in configs.seeds:
        seed_everything(seed=seed, deterministic=True)
        #logits = []
        #labels = []
        confidence = []
        distance = []
        names = []
        for batch in dataloader:
            try:
                data, atom_array, data_error_message = batch[0]

                if len(data_error_message) > 0:
                    logger.info(data_error_message)
                    with open(
                        opjoin(runner.error_dir, f"{data['sample_name']}.txt"),
                        "w",
                    ) as f:
                        f.write(data_error_message)
                    continue

                sample_name = data["sample_name"]
                logger.info(
                    (
                        f"[Rank {DIST_WRAPPER.rank} ({data['sample_index'] + 1}/{num_data})] {sample_name}: "
                        f"N_asym {data['N_asym'].item()}, N_token {data['N_token'].item()}, "
                        f"N_atom {data['N_atom'].item()}, N_msa {data['N_msa'].item()}"
                    )
                )

                prediction = runner.predict(data)
                # runner.dumper.dump(
                #     dataset_name="",
                #     pdb_id=sample_name,
                #     seed=seed,
                #     pred_dict=prediction,
                #     atom_array=atom_array,
                #     entity_poly_type=data["entity_poly_type"],
                # )
                # logit = prediction['binder']
                # print('logits',logit)
 
            
                print(data["entity_poly_type"])
                print(prediction.keys())
                print(prediction['summary_confidence'])
                print(prediction['coordinate'].shape)

                pred_atom_array = copy.deepcopy(atom_array)
                pred_pose = prediction['coordinate'].cpu().numpy()
                pred_atom_array.coord = pred_pose.squeeze(0)

                # Debug: Print available chain IDs
                print(f"Available chain IDs in pred_atom_array: {np.unique(pred_atom_array.label_asym_id)}")
                
                # Find DHHC motif cysteine in chain A (try different chain IDs)
                chain_a_cys = None
                cys_chain_id = None
                for chain_id in ['A']:
                    print(f"Trying to find DHHC in chain {chain_id}")
                    chain_a_cys = find_dhhc_motif(pred_atom_array, chain_id)
                    if chain_a_cys is not None:
                        cys_chain_id = chain_id
                        break
                
                # Find P1L residue in chain B (try different chain IDs)
                chain_b_p1l = None
                p1l_chain_id = None
                for chain_id in ['B']:
                    print(f"Trying to find P1L in chain {chain_id}")
                    chain_b_p1l = find_p1l_residue(pred_atom_array, chain_id)
                    if chain_b_p1l is not None:
                        p1l_chain_id = chain_id
                        break
                
                if chain_a_cys is not None and chain_b_p1l is not None:
                    # Calculate distance between cysteine sulfur and P1L SG
                    atom_distance = calculate_atom_distance(
                        pred_atom_array, 
                        chain1=cys_chain_id, res1=chain_a_cys[0], res_name1=chain_a_cys[1], atom1=chain_a_cys[2],
                        chain2=p1l_chain_id, res2=chain_b_p1l[0], res_name2=chain_b_p1l[1], atom2=chain_b_p1l[2]
                    )
                    print('atom_distance', atom_distance)
                    distance.append(atom_distance)
                    logger.info(f"[Rank {DIST_WRAPPER.rank}] {sample_name} 'distance': {atom_distance:.3f} Å")
                else:
                    print("Could not find DHHC motif or P1L residue")
                    distance.append(None)

                #logits.append(prediction['binder'].cpu().numpy())
                names.append(sample_name)
                confidence.append(prediction['summary_confidence'][0])

                logger.info(
                    f"[Rank {DIST_WRAPPER.rank}] {data['sample_name']} succeeded.\n"   
                )

            except Exception as e:
                error_message = f"[Rank {DIST_WRAPPER.rank}]{data['sample_name']} {e}:\n{traceback.format_exc()}"
                logger.info(error_message)
                # Save error info
                if opexists(
                    error_path := opjoin(runner.error_dir, f"{sample_name}.txt")
                ):
                    os.remove(error_path)
                with open(error_path, "w") as f:
                    f.write(error_message)
                if hasattr(torch.cuda, "empty_cache"):
                    torch.cuda.empty_cache()

    #torch.save(logits, os.path.join(configs.dump_dir, f'logits_{runname}.pt'))
    torch.save(names, os.path.join(configs.dump_dir, f'names_{runname}.pt'))
    torch.save(confidence, os.path.join(configs.dump_dir, f'confidence_{runname}.pt'))
    torch.save(distance, os.path.join(configs.dump_dir, f'distance_{runname}.pt'))
    
if __name__ == "__main__":
    LOG_FORMAT = "%(asctime)s,%(msecs)-3d %(levelname)-8s [%(filename)s:%(lineno)s %(funcName)s] %(message)s"
    logging.basicConfig(
        format=LOG_FORMAT,
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        filemode="w",
    )
    configs = {**configs_base, **{"data": data_configs}, **inference_configs}
    configs = parse_configs(
        configs=configs,
        arg_str=parse_sys_args(),
        fill_required_with_null=True,
    )
    main(configs)
