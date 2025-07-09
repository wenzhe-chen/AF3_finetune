#!/usr/bin/env python3
"""
Example usage of ConfidenceHead with DeepSetTransformerPooling for classification
"""

import torch
from protenix.model.modules.confidence import ConfidenceHead

def example_classifier_usage():
    """Example of using ConfidenceHead with classifier_head=True"""
    
    # Initialize the confidence head with classifier enabled
    confidence_head = ConfidenceHead(
        n_blocks=4,
        c_s=384,
        c_z=128,
        c_s_inputs=449,
        classifier_head=True  # Enable classification
    )
    
    # Example input dimensions
    N_tokens = 100
    N_atoms = 500
    N_sample = 2
    c_s = 384
    c_z = 128
    c_s_inputs = 449
    
    # Create dummy input tensors
    s_inputs = torch.randn(N_tokens, c_s_inputs)
    s_trunk = torch.randn(N_tokens, c_s)
    z_trunk = torch.randn(N_tokens, N_tokens, c_z)
    pair_mask = torch.ones(N_tokens, N_tokens, dtype=torch.bool)
    x_pred_coords = torch.randn(N_sample, N_atoms, 3)
    
    # Create input feature dictionary
    input_feature_dict = {
        "distogram_rep_atom_mask": torch.ones(N_atoms, dtype=torch.bool),
        "atom_to_token_idx": torch.randint(0, N_tokens, (N_atoms,)),
        "atom_to_tokatom_idx": torch.randint(0, 20, (N_atoms,)),  # max_atoms_per_token=20
    }
    
    # Forward pass
    outputs = confidence_head(
        input_feature_dict=input_feature_dict,
        s_inputs=s_inputs,
        s_trunk=s_trunk,
        z_trunk=z_trunk,
        pair_mask=pair_mask,
        x_pred_coords=x_pred_coords,
    )
    
    # Unpack outputs
    plddt_preds, pae_preds, pde_preds, resolved_preds, classification_scores = outputs
    
    print(f"Output shapes:")
    print(f"  plddt_preds: {plddt_preds.shape}")
    print(f"  pae_preds: {pae_preds.shape}")
    print(f"  pde_preds: {pde_preds.shape}")
    print(f"  resolved_preds: {resolved_preds.shape}")
    print(f"  classification_scores: {classification_scores.shape}")
    
    print(f"\nClassification scores: {classification_scores}")
    
    return outputs

def example_regular_usage():
    """Example of using ConfidenceHead without classifier (original behavior)"""
    
    # Initialize the confidence head without classifier
    confidence_head = ConfidenceHead(
        n_blocks=4,
        c_s=384,
        c_z=128,
        c_s_inputs=449,
        classifier_head=False  # Disable classification
    )
    
    # Example input dimensions
    N_tokens = 100
    N_atoms = 500
    N_sample = 2
    c_s = 384
    c_z = 128
    c_s_inputs = 449
    
    # Create dummy input tensors
    s_inputs = torch.randn(N_tokens, c_s_inputs)
    s_trunk = torch.randn(N_tokens, c_s)
    z_trunk = torch.randn(N_tokens, N_tokens, c_z)
    pair_mask = torch.ones(N_tokens, N_tokens, dtype=torch.bool)
    x_pred_coords = torch.randn(N_sample, N_atoms, 3)
    
    # Create input feature dictionary
    input_feature_dict = {
        "distogram_rep_atom_mask": torch.ones(N_atoms, dtype=torch.bool),
        "atom_to_token_idx": torch.randint(0, N_tokens, (N_atoms,)),
        "atom_to_tokatom_idx": torch.randint(0, 20, (N_atoms,)),
    }
    
    # Forward pass
    outputs = confidence_head(
        input_feature_dict=input_feature_dict,
        s_inputs=s_inputs,
        s_trunk=s_trunk,
        z_trunk=z_trunk,
        pair_mask=pair_mask,
        x_pred_coords=x_pred_coords,
    )
    
    # Unpack outputs
    plddt_preds, pae_preds, pde_preds, resolved_preds = outputs
    
    print(f"Output shapes (without classifier):")
    print(f"  plddt_preds: {plddt_preds.shape}")
    print(f"  pae_preds: {pae_preds.shape}")
    print(f"  pde_preds: {pde_preds.shape}")
    print(f"  resolved_preds: {resolved_preds.shape}")
    
    return outputs

if __name__ == "__main__":
    print("=== Example with Classifier ===")
    example_classifier_usage()
    
    print("\n=== Example without Classifier ===")
    example_regular_usage() 