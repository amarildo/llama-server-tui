# 🦙 LLamaLauncherTui

> A keyboard-driven Textual User Interface (TUI) launcher for `llama-server`.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![UI: Textual](https://img.shields.io/badge/UI-Textual-orange.svg)](https://github.com/Textualize/textual)
[![Hardware: NVIDIA RTX 4090 Optimized](https://img.shields.io/badge/RTX%204090-Optimized-green.svg)](#)

LLamaLauncherTui is a keyboard-driven terminal interface designed to simplify launching and managing `llama-server` configurations. Calibrated with default settings for the **NVIDIA RTX 4090 (24GB VRAM)** running **Qwen 3.6 (27B) UD-Q4_K_XL**, it allows configuring hardware, context, cache, HTTP, and sampler options in a single consolidated interface.

```
  ╦  ╦  ╔═╗ ╔╦╗ ╔═╗   ╦  ╔═╗ ╦ ╦ ╔╗╔ ╔═╗ ╦ ╦ ╔═╗ ╦═╗
  ║  ║  ╠═╣ ║║║ ╠═╣   ║  ╠═╣ ║ ║ ║║║ ║   ╠═╣ ║╣  ╠╦╝
  ╩═╝╩═╝╩ ╩ ╩ ╩ ╩ ╩   ╩═╝╩ ╩ ╚═╝ ╝╚╝ ╚═╝ ╩ ╩ ╚═╝ ╩╚═
```

---

## Features

**Gruvbox Styling:** A dark theme interface styled with custom HSL color tokens, compatible with other terminal themes (such as Dracula).
**Speculative Decoding (MTP) Support:** Full, native configuration parameters for Qwen 3.6 Multi-Token Prediction (MTP) draft-heads, allowing you to fine-tune `spec-draft-n-max` and probabilities.
**Smart Prefix Cache Re-use:** Configurable dynamic prefix prompt caching (`--cache-prompt`) and re-use token bounds (`--cache-reuse`), boosting multi-turn agent chat throughput by up to 22%.
**RTX 4090 Hardware Presets:** Pre-calibrated advanced VRAM configuration keys, including `MERGE_QKV`, `MERGE_EXPERTS` (expert matrix merging to conserve VRAM), and smart context checkpoints for massive (>100K) sequences.
**Config Toggle Persistence:** Safely uncheck any hardware or sampling parameter inside the UI. Your enabled/disabled states are stored persistently inside `config.json` via a strict `"DISABLED_FIELDS"` key.
**Integrated Modal File Browser:** Easily browse and select models (`.gguf`), servers, custom templates, and visual projectors with a single click.

---

## Quick Start

### 1. Requirements

Ensure you have Python 3.8+ and Textual installed:
```bash
pip install textual
```

*Note: The launcher does not require any other external dependencies.*

### 2. Setup & Installation

Clone or copy the project into a directory of your choice:
```bash
mkdir -p ~/LLamaLauncherTui
cd ~/LLamaLauncherTui
```

### 3. Run the Launcher

Launch the interactive configuration TUI using the wrapper script:
```bash
./llama-server-start
```
Or run the Python file directly:
```bash
python3 llama-server-tui.py
```

---

## Calibration for RTX 4090 & Qwen 3.6 MTP

To achieve maximum text generation speeds (**90+ tokens/second**), my presets are configured as follows:

* **Flash Attention:** Set to `on` (drastically lowers VRAM footprint).
* **KV Cache Quantization:** `CACHE_K` and `CACHE_V` set to `q8_0` (halves context RAM usage with zero quality loss).
* **Speculative Decoding:** Enabled via `SPEC_TYPE` set to `draft-mtp` with a `draft-n-max` of `3`.
* **DRY & XTC Samplers (Persistent Toggle):** 
  * While DRY and XTC provide excellent writing diversity, warping logit probabilities causes speculative draft tokens to be rejected by the main model, dropping speeds from **~90 t/s down to ~65 t/s**.
  * By default, DRY and XTC are **unchecked (disabled)** in this launcher to guarantee maximum throughput out of the box. Simply check them inside the **Advanced Samplers** collapsible if you prioritize creative output quality over raw execution speeds.

---

## Configuration Structure

The launcher stores configurations locally in `config.json`. A typical safe structure looks like:

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

---

## License

This project is licensed under the permissive **MIT License**. See the `LICENSE` file for details.
