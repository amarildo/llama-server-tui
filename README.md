# llama-server-tui

I built this keyboard-driven terminal UI launcher because I got tired of writing, maintaining, and copy-pasting 50-line `llama-server` shell commands in my terminal every time I wanted to switch models, adjust speculative drafting, or configure KV cache parameters.

It is a single-script, dependency-free (except for `textual`) dashboard that maps standard JSON configs directly to official `llama.cpp` server processes.

```
  ╦  ╦  ╔═╗ ╔╦╗ ╔═╗   ╦  ╔═╗ ╦ ╦ ╔╗╔ ╔═╗ ╦ ╦ ╔═╗ ╦═╗
  ║  ║  ╠═╣ ║║║ ╠═╣   ║  ╠═╣ ║ ║ ║║║ ║   ╠═╣ ║╣  ╠╦╝
  ╩═╝╩═╝╩ ╩ ╩ ╩ ╩ ╩   ╩═╝╩ ╩ ╚═╝ ╝╚╝ ╚═╝ ╩ ╩ ╚═╝ ╩╚═
```

## Key Highlights

- **Keyboard-Driven Efficiency**: Fast cursor navigation, checkboxes, and inline directory selectors. Designed to run smoothly inside TMUX or terminal splits without taking your hands off the keyboard.
- **Zero UI Thread Locks**: File system navigation and GGUF header parsing (block counts, architecture type, KV channels) are executed off-thread in background workers, ensuring the UI remains highly responsive even when loading massive >15GB models.
- **Resilient Self-Bootstrapping**: On the first execution, the script dynamically creates its own `./profiles/` directory and writes a generic, clean default preset based on your local machine standards.
- **Zero Corporate Bloat**: Written in pure, clean Python utilizing standard Textual widgets and the Python standard library.

## Installation and Setup

### Prerequisites

Ensure you have Python 3.8+ and Textual installed:

```bash
pip install textual
```

### Running the Launcher (Zero-Config)

You do **not** need to manually copy, create, or edit any configuration files. The application self-bootstraps completely. Simply run the launcher:

```bash
python3 llama-server-tui.py
```

Alternatively, you can run the provided wrapper script:

```bash
./llama-server-start
```

### First-Run Self-Bootstrapping
When launched for the first time, the launcher automatically:
1. Creates a `./profiles/` directory relative to the script's folder.
2. Dynamically generates a generic, high-performance `default.json` preset pre-calibrated for speculative decoding.
3. Spawns the dashboard, where you can **interactively browse and select** your local `llama-server` binary executable and GGUF model paths using the built-in keyboard-driven directory picker.


## Default Calibration (Qwen 3.6 27B MTP CUDA-On)

The launcher ships with pre-calibrated, high-performance default configurations tailored for speculative decoding setups on a single GPU (specifically optimized for an RTX 4090 running Qwen 3.6 27B UD-Q4_K_XL base and a speculative draft model with prompt caching):

- **High-Throughput Speculative Engine (`draft-mtp`)**: Pre-configured with Multi-Token Prediction (MTP) draft max count set to `3` and min probability threshold `0.15` for extremely high inference speeds (~90-100+ tokens/second).
- **Quantized KV Cache (`q8_0` / `q8_0`)**: Quantizes context keys and values to 8-bit to ensure optimal VRAM consumption even under large context windows.
- **CUDA Graphs & Flash Attention**: Combines Flash Attention (`on`) with active CUDA Graphs (disabled graphs: `off`) to leverage modern Ada Lovelace kernel optimizations for peak generation throughput.
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
    "SWA_FULL": "off",
    "GGML_CUDA_DISABLE_GRAPHS": "off",
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
