#!/usr/bin/env python3
"""
Simple script to load DHHC analysis results and export to CSV.
"""

import torch
import pandas as pd
from pathlib import Path

def load_and_export_dhhc_results():
    """
    Load DHHC analysis results and export to CSV.
    """
    base_path = Path("/home/fs01/wc648/protenix/output")
    
    # File paths
    confidence_file = base_path / "confidence_exmaple_DHHC.pt"
    distance_file = base_path / "distance_exmaple_DHHC.pt"
    names_file = base_path / "names_exmaple_DHHC.pt"
    
    # Check if files exist
    for file_path in [confidence_file, distance_file, names_file]:
        if not file_path.exists():
            print(f"Error: File {file_path} does not exist!")
            return
    
    # Load the data
    try:
        confidence_data = torch.load(confidence_file, map_location='cpu')
        distance_data = torch.load(distance_file, map_location='cpu')
        names_data = torch.load(names_file, map_location='cpu')
        
        print(f"Loaded data:")
        print(f"  Names: {len(names_data)} entries")
        print(f"  Distances: {len(distance_data)} entries")
        print(f"  Confidence: {len(confidence_data)} entries")
        
        # Debug: Show the structure of confidence data
        if len(confidence_data) > 0:
            print(f"\nConfidence data structure:")
            print(f"  Type: {type(confidence_data[0])}")
            if isinstance(confidence_data[0], dict):
                print(f"  Keys: {list(confidence_data[0].keys())}")
                for key, value in confidence_data[0].items():
                    print(f"    {key}: {type(value)} - {value}")
            else:
                print(f"  Value: {confidence_data[0]}")
                if hasattr(confidence_data[0], 'shape'):
                    print(f"  Shape: {confidence_data[0].shape}")
        
        # Check if all lists have the same length
        if len(names_data) != len(distance_data) or len(names_data) != len(confidence_data):
            print("Warning: Data lists have different lengths!")
            min_length = min(len(names_data), len(distance_data), len(confidence_data))
            names_data = names_data[:min_length]
            distance_data = distance_data[:min_length]
            confidence_data = confidence_data[:min_length]
            print(f"Truncated to {min_length} entries")
        
        # Create DataFrame
        df_data = []
        
        for i in range(len(names_data)):
            row = {'name': names_data[i]}
            
            # Add distance
            if i < len(distance_data) and distance_data[i] is not None:
                row['distance'] = float(distance_data[i])
            else:
                row['distance'] = None
            
            # Add confidence keys
            if i < len(confidence_data) and confidence_data[i] is not None:
                confidence_dict = confidence_data[i]
                if isinstance(confidence_dict, dict):
                    for key, value in confidence_dict.items():
                        # Convert tensor to appropriate format
                        if hasattr(value, 'item'):
                            try:
                                # Try to convert to scalar first
                                row[key] = float(value.item())
                            except:
                                # If it's a multi-element tensor, convert to list
                                if hasattr(value, 'tolist'):
                                    row[key] = str(value.tolist())
                                else:
                                    row[key] = str(value)
                        else:
                            row[key] = value
                else:
                    # If confidence is not a dict, just add it as 'confidence'
                    if hasattr(confidence_dict, 'item'):
                        try:
                            row['confidence'] = float(confidence_dict.item())
                        except:
                            if hasattr(confidence_dict, 'tolist'):
                                row['confidence'] = str(confidence_dict.tolist())
                            else:
                                row['confidence'] = str(confidence_dict)
                    else:
                        row['confidence'] = confidence_dict
            else:
                # Add empty confidence columns if we have a sample confidence dict
                if len(confidence_data) > 0 and confidence_data[0] is not None and isinstance(confidence_data[0], dict):
                    for key in confidence_data[0].keys():
                        row[key] = None
            
            df_data.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(df_data)
        
        # Print column information
        print(f"\nDataFrame shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Show first few rows
        print(f"\nFirst few rows:")
        print(df.head())
        
        # Export to CSV
        output_path = base_path / "dhhc_results.csv"
        df.to_csv(output_path, index=False)
        print(f"\nExported to: {output_path}")
        
        return df
        
    except Exception as e:
        print(f"Error loading or processing data: {e}")
        return None

if __name__ == "__main__":
    df = load_and_export_dhhc_results()
    if df is not None:
        print(f"\nSuccessfully exported {len(df)} rows to CSV")
    else:
        print("Failed to export data")
