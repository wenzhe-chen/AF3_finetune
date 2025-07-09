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

import json
import logging
import time
import traceback
import warnings
from typing import Any, Callable, Optional, Union, Mapping
from pathlib import Path
import os



import torch
from biotite.structure import AtomArray
from torch.utils.data import DataLoader, Dataset, DistributedSampler
import pandas as pd


from protenix.data.data_pipeline import DataPipeline
from protenix.data.json_to_feature import SampleDictToFeatures
from protenix.data.msa_featurizer import InferenceMSAFeaturizer
from protenix.data.utils import data_type_transform, make_dummy_feature
from protenix.utils.distributed import DIST_WRAPPER
from protenix.utils.torch_utils import dict_to_tensor
from protenix.data.dataset import SequenceClassificationDataset

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", module="biotite")

PANDAS_NA_VALUES = [
    "",
    "#N/A",
    "#N/A N/A",
    "#NA",
    "-1.#IND",
    "-1.#QNAN",
    "-NaN",
    "-nan",
    "1.#IND",
    "1.#QNAN",
    "<NA>",
    "N/A",
    # "NA",
    "NULL",
    "NaN",
    "n/a",
    "nan",
    "null",
]

def get_inference_dataloader(configs: Any) -> DataLoader:
    """
    Creates and returns a DataLoader for inference using the InferenceDataset.

    Args:
        configs: A configuration object containing the necessary parameters for the DataLoader.

    Returns:
        A DataLoader object configured for inference.
    """ 
    data_config = configs.data

    # not implement for multiple datasets
    config_dict = data_config['classifier_table']

    #print('config_dict',config_dict)

    inference_dataset = InferenceDataset(
        input_json_path=configs.input_json_path,
        dump_dir=configs.dump_dir,
        use_msa=configs.use_msa,
        precomputed_msa_dir=config_dict["base_info"]["precomputed_msa_dir"],
        msa_save_dir=config_dict["base_info"]["msa_save_dir"],
        msa_search_tool=config_dict["base_info"]["msa_search_tool"],
        msa_pairing_db=config_dict["base_info"]["msa_pairing_db"],
        msa_pairing_db_fpath=config_dict["base_info"]["msa_pairing_db_fpath"],
        msa_non_pairing_db_fpath=config_dict["base_info"]["msa_non_pairing_db_fpath"],
        train_classifier_by_inference=configs.train_classifier_by_inference,
    )

    sampler = DistributedSampler(
        dataset=inference_dataset,
        num_replicas=DIST_WRAPPER.world_size,
        rank=DIST_WRAPPER.rank,
        shuffle=False,
    )
    dataloader = DataLoader(
        dataset=inference_dataset,
        batch_size=1,
        sampler=sampler,
        collate_fn=lambda batch: batch,
        num_workers=configs.num_workers,
    )
    return dataloader


class InferenceDataset(Dataset):
    def __init__(
        self,
        input_json_path: str,
        dump_dir: str,
        use_msa: bool = True,
        **kwargs,
    ) -> None:

        self.input_json_path = input_json_path
        self.dump_dir = dump_dir
        self.use_msa = use_msa

        self.precomputed_msa_dir = kwargs.get("precomputed_msa_dir", None)
        self.msa_save_dir = kwargs.get("msa_save_dir", "./searched_msa")
        self.msa_search_tool = kwargs.get("msa_search_tool", "jackhmmer")
        self.msa_pairing_db = kwargs.get("msa_pairing_db", "uniprot")
        self.msa_pairing_db_fpath = kwargs.get("msa_pairing_db_fpath", "/home/fs01/wc648/RoseTTAFold-All-Atom/uniprot/uniprot_sprot.fasta")
        self.msa_non_pairing_db_fpath = kwargs.get("msa_non_pairing_db_fpath", "/home/fs01/wc648/RoseTTAFold-All-Atom/mgnify/mgy_clusters_2018_12.fa")
        self.train_classifier_by_inference = kwargs.get("train_classifier_by_inference", False)

        if self.input_json_path.endswith(".json"):
            with open(self.input_json_path, "r") as f:
                self.inputs = json.load(f)
        elif self.input_json_path.endswith(".csv"):
            #self.inputs = pd.read_csv(self.input_json_path)
            self.inputs = self.load_inputs(self.input_json_path)
    
    def load_inputs(self, indices_fpath: Union[str, Path]) -> list[dict]:
        """
        Reads and processes a list of indices from a CSV file.

        Args:
            indices_fpath: Path to the CSV file containing the indices.

        Returns:
            A List of dicts containing the processed indices.
        """
        indices_list = pd.read_csv(indices_fpath, na_values=PANDAS_NA_VALUES, keep_default_na=False, dtype=str)
        num_data = len(indices_list)
        logger.info(f"#Rows in indices list: {num_data}")
        inputs = []
        for idx in range(num_data):
            #print(indices_list.iloc[idx])
            input = {}
            input["name"] = indices_list.iloc[idx]["name"]
            input["model_seed"] = []
            input["assembly_id"] = "1"
            input["label"] = indices_list.iloc[idx]["label"]
            sequence_list = indices_list.iloc[idx]["sequences"].split(":")
            sequences = []
            for j in range(len(sequence_list)):
                chain_dict = {}
                if len(sequence_list[j]) > 50:
                    chain_dict["proteinChain"] = {
                        "sequence": sequence_list[j],
                        "count": 1,
                        "msa": {
                            "precomputed_msa_dir": os.path.join(self.precomputed_msa_dir, "1"),
                            "search_tool": self.msa_search_tool,
                            "pairing_db": self.msa_pairing_db,
                            "pairing_db_fpath": self.msa_pairing_db_fpath,
                            "non_pairing_db_fpath": self.msa_non_pairing_db_fpath,
                            "msa_save_dir": self.msa_save_dir
                        }
                    }
                else:
                    chain_dict["proteinChain"] = {
                        "sequence": sequence_list[j],
                        "count": 1,
                        "msa": {
                            "precomputed_msa_dir": os.path.join(self.precomputed_msa_dir, "2"),
                            "search_tool": self.msa_search_tool,
                            "pairing_db": self.msa_pairing_db,
                            "pairing_db_fpath": self.msa_pairing_db_fpath,
                            "non_pairing_db_fpath": self.msa_non_pairing_db_fpath,
                            "msa_save_dir": self.msa_save_dir
                        }
                    }
                sequences.append(chain_dict)
            input["sequences"] = sequences

            inputs.append(input)
        return inputs

    def process_one(
        self,
        single_sample_dict: Mapping[str, Any],
    ) -> tuple[dict[str, torch.Tensor], AtomArray, dict[str, float]]:
        """
        Processes a single sample from the input JSON to generate features and statistics.

        Args:
            single_sample_dict: A dictionary containing the sample data.

        Returns:
            A tuple containing:
                - A dictionary of features.
                - An AtomArray object.
                - A dictionary of time tracking statistics.
        """
        # general features
        t0 = time.time()
        sample2feat = SampleDictToFeatures(
            single_sample_dict,
        )
        features_dict, atom_array, token_array = sample2feat.get_feature_dict()
        features_dict["distogram_rep_atom_mask"] = torch.Tensor(
            atom_array.distogram_rep_atom_mask
        ).long()
        entity_poly_type = sample2feat.entity_poly_type
        t1 = time.time()

        # Msa features
        entity_to_asym_id = DataPipeline.get_label_entity_id_to_asym_id_int(atom_array)
        msa_features = (
            InferenceMSAFeaturizer.make_msa_feature(
                bioassembly=single_sample_dict["sequences"],
                entity_to_asym_id=entity_to_asym_id,
                token_array=token_array,
                atom_array=atom_array,
            )
            if self.use_msa
            else {}
        )

        # Make dummy features for not implemented features
        dummy_feats = ["template"]
        if len(msa_features) == 0:
            dummy_feats.append("msa")
        else:
            msa_features = dict_to_tensor(msa_features)
            features_dict.update(msa_features)
        features_dict = make_dummy_feature(
            features_dict=features_dict,
            dummy_feats=dummy_feats,
        )

        # Transform to right data type
        feat = data_type_transform(feat_or_label_dict=features_dict)

        t2 = time.time()

        data = {}
        data["input_feature_dict"] = feat
        if self.train_classifier_by_inference:
            data["label_dict"] = int(single_sample_dict["label"])

        # Add dimension related items
        N_token = feat["token_index"].shape[0]
        N_atom = feat["atom_to_token_idx"].shape[0]
        N_msa = feat["msa"].shape[0]

        stats = {}
        for mol_type in ["ligand", "protein", "dna", "rna"]:
            mol_type_mask = feat[f"is_{mol_type}"].bool()
            stats[f"{mol_type}/atom"] = int(mol_type_mask.sum(dim=-1).item())
            stats[f"{mol_type}/token"] = len(
                torch.unique(feat["atom_to_token_idx"][mol_type_mask])
            )

        N_asym = len(torch.unique(data["input_feature_dict"]["asym_id"]))
        data.update(
            {
                "N_asym": torch.tensor([N_asym]),
                "N_token": torch.tensor([N_token]),
                "N_atom": torch.tensor([N_atom]),
                "N_msa": torch.tensor([N_msa]),
            }
        )

        def formatted_key(key):
            type_, unit = key.split("/")
            if type_ == "protein":
                type_ = "prot"
            elif type_ == "ligand":
                type_ = "lig"
            else:
                pass
            return f"N_{type_}_{unit}"

        data.update(
            {
                formatted_key(k): torch.tensor([stats[k]])
                for k in [
                    "protein/atom",
                    "ligand/atom",
                    "dna/atom",
                    "rna/atom",
                    "protein/token",
                    "ligand/token",
                    "dna/token",
                    "rna/token",
                ]
            }
        )
        data.update({"entity_poly_type": entity_poly_type})
        t3 = time.time()
        time_tracker = {
            "crop": t1 - t0,
            "featurizer": t2 - t1,
            "added_feature": t3 - t2,
        }

        return data, atom_array, time_tracker

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, index: int) -> tuple[dict[str, torch.Tensor], AtomArray, str]:
        try:
            single_sample_dict = self.inputs[index]
            sample_name = single_sample_dict["name"]
            logger.info(f"Featurizing {sample_name}...")

            data, atom_array, _ = self.process_one(
                single_sample_dict=single_sample_dict
            )
            error_message = ""
        except Exception as e:
            data, atom_array = {}, None
            error_message = f"{e}:\n{traceback.format_exc()}"
        data["sample_name"] = single_sample_dict["name"]
        data["sample_index"] = index
        data["atom_array"] = atom_array
        return data, atom_array, error_message
