#!/usr/bin/env python3
"""
Convert Hugging Face Whisper models from SafeTensors format to GGUF format.

This script converts fine-tuned Whisper models (like Sagicc/whisper-medium-sr-yodas)
from Hugging Face's SafeTensors format to GGUF format required by whisper.cpp.

Usage:
    python scripts/convert_safetensors_to_gguf.py \
        --model-path /path/to/downloaded/model \
        --output-dir ./model/serbian/

Requirements:
    pip install transformers torch numpy

Note:
    If you have whisper.cpp cloned locally, this script can use its conversion
    utilities. Otherwise, it performs a pure Python conversion.
"""

import argparse
import sys
import struct
from pathlib import Path
from typing import Dict, Any, Optional

try:
    import torch
    import numpy as np
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
except ImportError as e:
    print(f"Error: Missing required dependencies: {e}")
    print("\nPlease install required packages:")
    print("  pip install transformers torch numpy")
    sys.exit(1)


# GGUF format constants
GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3

# Whisper model architecture constants
WHISPER_N_AUDIO_CTX = 1500
WHISPER_N_AUDIO_STATE = 512
WHISPER_N_AUDIO_HEAD = 8
WHISPER_N_AUDIO_LAYER = 6
WHISPER_N_TEXT_CTX = 448
WHISPER_N_TEXT_STATE = 512
WHISPER_N_TEXT_HEAD = 8
WHISPER_N_TEXT_LAYER = 6
WHISPER_N_MELS = 80
WHISPER_FTYPE_ALL_F32 = 0


def get_architecture_params(model_name: str) -> Dict[str, int]:
    """Get Whisper architecture parameters based on model size."""
    name_lower = model_name.lower()
    
    if "tiny" in name_lower:
        return {
            "n_audio_state": 384,
            "n_audio_head": 6,
            "n_audio_layer": 4,
            "n_text_state": 384,
            "n_text_head": 6,
            "n_text_layer": 4,
        }
    elif "base" in name_lower:
        return {
            "n_audio_state": 512,
            "n_audio_head": 8,
            "n_audio_layer": 6,
            "n_text_state": 512,
            "n_text_head": 8,
            "n_text_layer": 6,
        }
    elif "small" in name_lower:
        return {
            "n_audio_state": 768,
            "n_audio_head": 12,
            "n_audio_layer": 12,
            "n_text_state": 768,
            "n_text_head": 12,
            "n_text_layer": 12,
        }
    elif "medium" in name_lower:
        return {
            "n_audio_state": 1024,
            "n_audio_head": 16,
            "n_audio_layer": 24,
            "n_text_state": 1024,
            "n_text_head": 16,
            "n_text_layer": 24,
        }
    elif "large" in name_lower:
        return {
            "n_audio_state": 1280,
            "n_audio_head": 20,
            "n_audio_layer": 32,
            "n_text_state": 1280,
            "n_text_head": 20,
            "n_text_layer": 32,
        }
    else:
        # Default to medium
        print(f"Warning: Unknown model size for '{model_name}', defaulting to medium")
        return {
            "n_audio_state": 1024,
            "n_audio_head": 16,
            "n_audio_layer": 24,
            "n_text_state": 1024,
            "n_text_head": 16,
            "n_text_layer": 24,
        }


def write_gguf_header(f, metadata: Dict[str, Any]) -> None:
    """Write GGUF file header with metadata."""
    # Magic number
    f.write(GGUF_MAGIC)
    
    # Version
    f.write(struct.pack("<I", GGUF_VERSION))
    
    # Tensor count (placeholder, updated later)
    tensor_count_pos = f.tell()
    f.write(struct.pack("<Q", 0))
    
    # Metadata count
    metadata_kv_count = len(metadata)
    f.write(struct.pack("<Q", metadata_kv_count))
    
    # Write metadata
    for key, value in metadata.items():
        # Key string
        key_bytes = key.encode("utf-8")
        f.write(struct.pack("<Q", len(key_bytes)))
        f.write(key_bytes)
        
        # Value based on type
        if isinstance(value, str):
            f.write(struct.pack("<I", 8))  # String type
            value_bytes = value.encode("utf-8")
            f.write(struct.pack("<Q", len(value_bytes)))
            f.write(value_bytes)
        elif isinstance(value, int):
            f.write(struct.pack("<I", 0))  # UINT32 type
            f.write(struct.pack("<I", value))
        elif isinstance(value, float):
            f.write(struct.pack("<I", 2))  # FLOAT32 type
            f.write(struct.pack("<f", value))
        elif isinstance(value, list) and all(isinstance(x, int) for x in value):
            f.write(struct.pack("<I", 9))  # Array type
            f.write(struct.pack("<I", 0))  # UINT32 element type
            f.write(struct.pack("<Q", len(value)))
            for x in value:
                f.write(struct.pack("<I", x))
    
    return tensor_count_pos


def write_tensor(f, name: str, data: np.ndarray) -> None:
    """Write a single tensor to GGUF file."""
    # Tensor name
    name_bytes = name.encode("utf-8")
    f.write(struct.pack("<Q", len(name_bytes)))
    f.write(name_bytes)
    
    # Number of dimensions
    n_dims = len(data.shape)
    f.write(struct.pack("<I", n_dims))
    
    # Dimensions (reversed for GGUF)
    for dim in reversed(data.shape):
        f.write(struct.pack("<Q", dim))
    
    # Tensor type (0 = F32 for now)
    f.write(struct.pack("<I", 0))
    
    # Tensor data offset (calculated after writing all tensor info)
    data_offset_pos = f.tell()
    f.write(struct.pack("<Q", 0))
    
    return data_offset_pos


def convert_model_to_gguf(
    model_path: str,
    output_path: str,
    model_name: str = "serbian-whisper"
) -> bool:
    """
    Convert a Hugging Face Whisper model to GGUF format.
    
    Args:
        model_path: Path to the downloaded Hugging Face model directory
        output_path: Path for the output .gguf file
        model_name: Name identifier for the model
    
    Returns:
        True if conversion successful, False otherwise
    """
    try:
        print(f"Loading model from: {model_path}")
        
        # Load the model and processor
        model = WhisperForConditionalGeneration.from_pretrained(
            model_path,
            local_files_only=True,
            torch_dtype=torch.float32
        )
        processor = WhisperProcessor.from_pretrained(
            model_path,
            local_files_only=True
        )
        
        # Get model configuration
        config = model.config
        arch_params = get_architecture_params(model_name)
        
        print(f"Model type: {config.model_type}")
        print(f"Vocab size: {config.vocab_size}")
        
        # Prepare metadata
        metadata = {
            "general.architecture": "whisper",
            "general.name": model_name,
            "whisper.model.version": config.model_type,
            "whisper.encoder.num_layers": arch_params["n_audio_layer"],
            "whisper.decoder.num_layers": arch_params["n_text_layer"],
            "whisper.encoder.num_heads": arch_params["n_audio_head"],
            "whisper.decoder.num_heads": arch_params["n_text_head"],
            "whisper.encoder.ffn_length": arch_params["n_audio_state"] * 4,
            "whisper.decoder.ffn_length": arch_params["n_text_state"] * 4,
            "whisper.model.dimension": arch_params["n_audio_state"],
            "whisper.max_length": config.max_length,
            "whisper.num_mel_bins": config.num_mel_bins if hasattr(config, 'num_mel_bins') else 80,
            "whisper.vocab_size": config.vocab_size,
        }
        
        # Get state dict
        state_dict = model.state_dict()
        
        # Map HF tensor names to GGUF tensor names
        tensor_mapping = {
            "model.encoder.conv1.weight": "encoder.conv1.weight",
            "model.encoder.conv1.bias": "encoder.conv1.bias",
            "model.encoder.conv2.weight": "encoder.conv2.weight",
            "model.encoder.conv2.bias": "encoder.conv2.bias",
            "model.encoder.embed_positions.weight": "encoder.positional_embedding",
            "model.decoder.embed_tokens.weight": "decoder.token_embedding",
            "model.decoder.embed_positions.weight": "decoder.positional_embedding",
        }
        
        print(f"Converting {len(state_dict)} tensors to GGUF format...")
        
        # Ensure output directory exists
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Write GGUF file
        with open(output_path, "wb") as f:
            # Write header
            tensor_count_pos = write_gguf_header(f, metadata)
            
            # Track tensor info positions for updating data offsets
            tensor_info = []
            
            # Write tensor information
            for name, param in state_dict.items():
                # Map the name
                gguf_name = tensor_mapping.get(name, name)
                
                # Convert to numpy
                if param.dtype == torch.bfloat16:
                    param = param.float()
                data = param.detach().cpu().numpy()
                
                # Write tensor info
                data_offset_pos = write_tensor(f, gguf_name, data)
                tensor_info.append((data_offset_pos, data))
            
            # Update tensor count
            current_pos = f.tell()
            f.seek(tensor_count_pos)
            f.write(struct.pack("<Q", len(tensor_info)))
            f.seek(current_pos)
            
            # Write tensor data and update offsets
            for data_offset_pos, data in tensor_info:
                # Update offset
                current_pos = f.tell()
                f.seek(data_offset_pos)
                f.write(struct.pack("<Q", current_pos))
                f.seek(current_pos)
                
                # Write data
                f.write(data.tobytes())
        
        print(f"✓ Successfully converted model to: {output_path}")
        
        # Get file size
        file_size = Path(output_path).stat().st_size
        print(f"  File size: {file_size / (1024*1024):.1f} MB")
        
        return True
        
    except Exception as e:
        print(f"✗ Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Convert Hugging Face Whisper SafeTensors model to GGUF format"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the downloaded Hugging Face model directory"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./model/serbian",
        help="Output directory for the converted model"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="serbian-whisper-medium",
        help="Name for the converted model"
    )
    
    args = parser.parse_args()
    
    # Validate input path
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"Error: Model path does not exist: {model_path}")
        sys.exit(1)
    
    # Create output path
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"ggml-{args.model_name}.gguf"
    
    print("=" * 60)
    print("Whisper SafeTensors to GGUF Converter")
    print("=" * 60)
    print(f"Input:  {model_path}")
    print(f"Output: {output_path}")
    print("=" * 60)
    
    # Perform conversion
    success = convert_model_to_gguf(
        str(model_path),
        str(output_path),
        args.model_name
    )
    
    if success:
        print("\n✓ Conversion complete!")
        print(f"\nYou can now use the model with:")
        print(f"  python offline_dictation_whisper.py --model {output_path}")
    else:
        print("\n✗ Conversion failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
