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

## Default Calibration (Qwen 3.6 27B MTP CUDA-On)

The launcher ships with pre-calibrated, high-performance default configurations tailored for speculative decoding setups on a single GPU (specifically optimized for an RTX 4090 running Qwen 3.6 27B UD-Q4_K_XL base and a speculative draft model with prompt caching):

- **High-Throughput Speculative Engine (`draft-mtp`)**: Pre-configured with Multi-Token Prediction (MTP) draft max count set to `3` and min probability threshold `0.15` for extremely high inference speeds (~90-100+ tokens/second).
- **Quantized KV Cache (`q8_0` / `q8_0`)**: Quantizes context keys and values to 8-bit to ensure optimal VRAM consumption even under large context windows.
- **CUDA Graphs & Flash Attention**: Combines Flash Attention (`on`) with graph-safe defaults (`GGML_CUDA_DISABLE_GRAPHS=1`) to guarantee perfect driver stability under Ada Lovelace architectures when running low-bit attention caches.
- **Optimal Memory Layouts**: Configures extremely wide batch processing sizes (`--batch-size 65536`) and micro-batching (`--ubatch-size 256`) to maximize pipeline throughput.
- **Jinja Reasoning**: Enables Jinja prompt template parsing (`on`) and pre-configures JSON template arguments (`preserve_thinking: true`) to natively retain thinking traces for agentic/reasoning models.

## Configuration File Structure

Your local `config.json` maps parameter keys directly to command-line flags. Below is the standard structure matching the pre-calibrated default profile:

```json
{
    "BIN": "~/llama.cpp/build/bin/llama-server",
    "MODEL": "~/models/Qwen3.6-27B-UD-Q4_K_XL.gguf",
    "HOST": "127.0.0.1",
    "PORT": "8080",
    "ALIAS": "qwen3.6-27b-mtp",
    "NGL": "99",
    "CTX": "85000",
    "FLASH_ATTN": "on",
    "CACHE_K": "q8_0",
    "CACHE_V": "q8_0",
    "SPEC_TYPE": "draft-mtp",
    "SPEC_MAX": "3",
    "SPEC_MIN": "0.15",
    "JINJA": "on",
    "WEBUI": "on",
    "SWA_FULL": "on",
    "GGML_CUDA_DISABLE_GRAPHS": "on",
    "DISABLED_FIELDS": [
        "API_KEY",
        "THREADS",
        "THREADS_BATCH",
        "BATCH_SIZE",
        "UBATCH_SIZE",
        "MLOCK",
        "SPLIT_MODE",
        "MAIN_GPU",
        "TENSOR_SPLIT",
        "NUMA",
        "GGML_CUDA_DISABLE_GRAPHS",
        "CTX_CHECKPOINTS",
        "SWA_FULL",
        "CACHE_RAM",
        "CONT_BATCHING",
        "CACHE_PROMPT",
        "CACHE_REUSE",
        "NO_MMAP",
        "THREADS_HTTP",
        "PREDICT",
        "SEED",
        "DRY_MULTIPLIER",
        "DRY_BASE",
        "DRY_ALLOWED_LENGTH",
        "DRY_PENALTY_LAST_N",
        "XTC_PROBABILITY",
        "XTC_THRESHOLD",
        "SPEC_DRAFT_N_MIN",
        "DRAFT_MAX",
        "DRAFT_P_MIN",
        "REASONING_FORMAT",
        "REASONING_BUDGET",
        "METRICS",
        "MMPROJ",
        "MMPROJ_OFFLOAD",
        "IMAGE_MIN_TOKENS",
        "IMAGE_MAX_TOKENS",
        "CHAT_TEMPLATE",
        "CHAT_TEMPLATE_FILE",
        "SKIP_CHAT_PARSING"
    ]
}
```

## License

MIT
