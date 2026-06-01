# llama-server-tui

This is a terminal UI launcher for llama-server. It simplifies configuring and executing llama-server without needing to manually copy-paste long, convoluted shell commands every time you change models or parameters.

The default presets are calibrated for running a speculative decoding setup on a single GPU (specifically tested on an RTX 4090 with Qwen 3.6 27B UD-Q4_K_XL using prompt caching).

```
  ╦  ╦  ╔═╗ ╔╦╗ ╔═╗   ╦  ╔═╗ ╦ ╦ ╔╗╔ ╔═╗ ╦ ╦ ╔═╗ ╦═╗
  ║  ║  ╠═╣ ║║║ ╠═╣   ║  ╠═╣ ║ ║ ║║║ ║   ╠═╣ ║╣  ╠╦╝
  ╩═╝╩═╝╩ ╩ ╩ ╩ ╩ ╩   ╩═╝╩ ╩ ╚═╝ ╝╚╝ ╚═╝ ╩ ╩ ╚═╝ ╩╚═
```

## How It Works

The launcher loads, displays, and saves configuration parameters from a local config.json file. When launching, it configures environment variables (such as disabling CUDA Graphs or enabling Unified Memory) and spawns llama-server as a subprocess.

* Gruvbox Palette: A clean, hardcoded styling matching dark terminal palettes.
* Parameter Configuration: Keyboard-driven toggles and fields for model performance parameters (like DRY, XTC, and Reasoning options).
* Directory Browser: Built-in minimal path picker to easily locate GGUF models and the server binary.
* Clean Architecture: Pure Python built on top of the Textual library. No bloated enterprise frameworks.

## Installation and Setup

### Prerequisites

Ensure you have Python 3.8+ and Textual installed:

```bash
pip install textual
```

### Setup Configuration

Before running the application, you must set up your local configuration:

1. Copy the example configuration file:
   ```bash
   cp config.json.example config.json
   ```
2. Edit config.json to match your local paths (specifically BIN for the llama-server binary path and MODEL for your default GGUF model path).

### Running the Launcher

Start the interface with:

```bash
python3 llama-server-tui.py
```

Alternatively, you can run the provided wrapper script:

```bash
./llama-server-start
```

## RTX 4090 & Qwen 3.6 MTP Presets

For speculative decoding setups aiming for maximum throughput (90+ tokens/second):
- Flash Attention: Enabled to minimize VRAM footprint.
- KV Cache Quantization: CACHE_K and CACHE_V quantized to q8_0.
- Speculative Decoding: Enabled with speculative type draft-mtp and draft max count of 3.
- Samplers: Disable sampler warping options like DRY and XTC in the UI to prevent spec draft throughput drops.

## Configuration File Structure

Your config.json maps parameter keys to command-line flags. Below is an example structure:

```json
{
    "BIN": "~/llama.cpp/build/bin/llama-server",
    "MODEL": "~/.cache/huggingface/hub/.../Qwen3.6-27B-UD-Q4_K_XL.gguf",
    "HOST": "127.0.0.1",
    "PORT": "8080",
    "FLASH_ATTN": "on",
    "CACHE_K": "q8_0",
    "CACHE_V": "q8_0",
    "SPEC_TYPE": "draft-mtp",
    "DISABLED_FIELDS": [
        "DRY_MULTIPLIER",
        "DRY_BASE",
        "DRY_ALLOWED_LENGTH",
        "DRY_PENALTY_LAST_N",
        "XTC_PROBABILITY",
        "XTC_THRESHOLD"
    ]
}
```

## License

MIT
