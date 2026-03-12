# SafeTensors to GGUF Conversion Script Specification

## Overview
This document specifies the conversion script needed to transform Hugging Face Whisper models from SafeTensors format to GGUF format compatible with whisper.cpp.

## Background
The `Sagicc/whisper-medium-sr-yodas` model (and other Hugging Face models) are distributed in SafeTensors format, but whisper.cpp requires GGUF format.

## Option 1: Using whisper.cpp's Conversion Tools (Recommended)

The most reliable approach is to use the official whisper.cpp conversion utilities.

### Setup Steps:
```bash
# 1. Clone whisper.cpp repository
git clone https://github.com/ggerganov/whisper.cpp.git
cd whisper.cpp

# 2. Install Python dependencies
pip install torch transformers numpy

# 3. Download the Serbian model
mkdir -p models/serbian
huggingface-cli download Sagicc/whisper-medium-sr-yodas \
    --local-dir models/serbian/whisper-medium-sr-yodas \
    --local-dir-use-symlinks False

# 4. Convert using whisper.cpp's convert script
python models/convert-h5-to-ggml.py \
    models/serbian/whisper-medium-sr-yodas \
    ./
```

### Known Issues:
- `convert-h5-to-ggml.py` expects PyTorch `.bin` files, not SafeTensors
- Need to convert SafeTensors to PyTorch format first, or use alternative method

## Option 2: Using Hugging Face to GGML Direct Conversion

### Python Script Approach:
```python
#!/usr/bin/env python3
"""Convert SafeTensors Whisper model to GGUF format."""

import torch
from transformers import WhisperForConditionalGeneration
import numpy as np
import struct
from pathlib import Path

def convert_to_gguf(model_path: str, output_path: str):
    # Load model
    model = WhisperForConditionalGeneration.from_pretrained(
        model_path,
        local_files_only=True,
        torch_dtype=torch.float32
    )
    
    # Get state dict
    state_dict = model.state_dict()
    
    # Write GGUF file (simplified structure)
    with open(output_path, 'wb') as f:
        # Write GGUF header
        f.write(b'GGUF')
        f.write(struct.pack('<I', 3))  # Version 3
        f.write(struct.pack('<Q', len(state_dict)))  # Tensor count
        f.write(struct.pack('<Q', 0))  # Metadata count (simplified)
        
        # Write tensors
        for name, param in state_dict.items():
            data = param.detach().cpu().numpy().astype(np.float32)
            
            # Write tensor name
            name_bytes = name.encode('utf-8')
            f.write(struct.pack('<Q', len(name_bytes)))
            f.write(name_bytes)
            
            # Write dimensions
            f.write(struct.pack('<I', len(data.shape)))
            for dim in reversed(data.shape):
                f.write(struct.pack('<Q', dim))
            
            # Write type (F32 = 0)
            f.write(struct.pack('<I', 0))
            
            # Write data offset and data
            f.write(struct.pack('<Q', f.tell() + 8))
            f.write(data.tobytes())
    
    print(f"Converted to: {output_path}")
```

## Option 3: Using Alternative Pre-Converted Models

Instead of converting, use models already in GGUF format:

```bash
# Check if community has already converted the model
# Search: https://huggingface.co/models?search=whisper%20gguf%20serbian

# Or use base whisper.cpp models with Serbian language flag:
wget https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.bin
```

## Recommended Approach for Your Project

Given the complexity of GGUF format, I recommend:

1. **Immediate Solution**: Use the existing `ggml-large-v3-turbo-q5_0.bin` model with `-l sr` flag
   - Add `--language sr` parameter to force Serbian
   - No model conversion needed

2. **Long-term Solution**: Clone whisper.cpp and use their conversion pipeline
   - More reliable than custom Python conversion
   - Handles all edge cases properly

## Integration Plan

### Modified Files:
1. `modules/config.py` - Add `LANGUAGE = 'sr'`
2. `offline_dictation_whisper.py` - Add `--language` CLI arg
3. `run_dictation.sh` - Auto-detect Serbian model or use language flag

### Usage:
```bash
# With existing model + language flag
python offline_dictation_whisper.py --model model/ggml-large-v3-turbo-q5_0.bin --language sr

# With converted Serbian model (future)
python offline_dictation_whisper.py --model model/serbian/ggml-whisper-medium-sr-yodas.gguf --language sr
```

## Next Steps
1. Decide on conversion approach
2. Implement language parameter support
3. Test with Serbian audio samples
