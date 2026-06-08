#!/usr/bin/env python3
import os
import time
import logging
import json
import shlex
import subprocess
import struct
from textual.app import App, ComposeResult
from textual.screen import ModalScreen
from textual import on
from textual.containers import VerticalScroll, Horizontal, Vertical, Container
from textual.widgets import Footer, Input, Button, Label, Collapsible, Checkbox, Static, Select
from textual.reactive import reactive
from pathlib import Path

CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(CONFIG_DIR, "config.json")

logging.basicConfig(filename=os.path.join(CONFIG_DIR, 'tui_debug.log'), level=logging.DEBUG, format='%(asctime)s %(message)s')

GGUF_CACHE = {}
import threading
GGUF_LOCK = threading.Lock()
PENDING_GGUF_LOADS = set()

def pre_parse_gguf_params(metadata: dict, model_path: str) -> dict:
    """Pre-parse architectural parameters from raw metadata to prevent real-time string scanning."""
    block_count = 32
    head_count_kv = 8
    embedding_length = 4096
    head_count = 32

    for k, v in metadata.items():
        kl = k.lower()
        try:
            if kl.endswith(".block_count"):
                block_count = int(v)
            elif kl.endswith(".attention.head_count_kv"):
                head_count_kv = int(v)
            elif kl.endswith(".embedding_length"):
                embedding_length = int(v)
            elif kl.endswith(".attention.head_count"):
                head_count = int(v)
        except (ValueError, TypeError):
            pass

    # Head dimension determination
    head_dim = 128
    if model_path and "gemma" in model_path.lower():
        head_dim = 256
    else:
        if head_count and embedding_length:
            calculated_dim = embedding_length // head_count
            if calculated_dim in (64, 80, 96, 128, 256):
                head_dim = calculated_dim

    return {
        "block_count": block_count,
        "head_count_kv": head_count_kv,
        "embedding_length": embedding_length,
        "head_count": head_count,
        "head_dim": head_dim
    }

def get_model_info(model_path: str, app=None) -> dict:
    if not model_path:
        return {
            "size_gib": 16.5,
            "metadata": {},
            "parsed": True,
            "params": {
                "block_count": 32,
                "head_count_kv": 8,
                "embedding_length": 4096,
                "head_count": 32,
                "head_dim": 128
            }
        }
    model_path = os.path.expanduser(model_path)

    with GGUF_LOCK:
        if model_path in GGUF_CACHE:
            return GGUF_CACHE[model_path]

        # Spawn background parser if app instance is provided
        if app is not None and model_path not in PENDING_GGUF_LOADS:
            if os.path.isfile(model_path):
                PENDING_GGUF_LOADS.add(model_path)
                app.run_worker(lambda: app._load_gguf_metadata_worker(model_path), thread=True)

    # Return responsive placeholder during background loading
    return {
        "size_gib": 16.5,
        "metadata": {},
        "parsed": False,
        "loading": True,
        "params": {
            "block_count": 32,
            "head_count_kv": 8,
            "embedding_length": 4096,
            "head_count": 32,
            "head_dim": 128
        }
    }

def truncate_path(path: str) -> str:
    if not path or len(path) <= 30:
        return path
    path = os.path.expanduser(path)
    parts = os.path.split(path)
    if parts[1]:
        return f".../{parts[1]}"
    return path

def parse_gguf_metadata(model_path: str) -> dict:
    metadata = {}
    if not model_path:
        return metadata
    model_path = os.path.expanduser(model_path)
    if not os.path.isfile(model_path):
        return metadata
    try:
        with open(model_path, "rb") as f:
            magic = f.read(4)
            if magic != b"GGUF":
                return metadata
            version = struct.unpack("<I", f.read(4))[0]
            if version not in (1, 2, 3):
                return metadata
            tensor_count = struct.unpack("<Q", f.read(8))[0]
            metadata_count = struct.unpack("<Q", f.read(8))[0]
            
            def read_str(file_obj):
                length = struct.unpack("<Q", file_obj.read(8))[0]
                return file_obj.read(length).decode("utf-8", errors="ignore")
                
            def read_val(file_obj, v_type):
                if v_type == 0: return struct.unpack("<B", file_obj.read(1))[0]
                elif v_type == 1: return struct.unpack("<b", file_obj.read(1))[0]
                elif v_type == 2: return struct.unpack("<H", file_obj.read(2))[0]
                elif v_type == 3: return struct.unpack("<h", file_obj.read(2))[0]
                elif v_type == 4: return struct.unpack("<I", file_obj.read(4))[0]
                elif v_type == 5: return struct.unpack("<i", file_obj.read(4))[0]
                elif v_type == 6: return struct.unpack("<f", file_obj.read(4))[0]
                elif v_type == 7: return struct.unpack("<?", file_obj.read(1))[0]
                elif v_type == 8: return read_str(file_obj)
                elif v_type == 9:
                    sub_type = struct.unpack("<I", file_obj.read(4))[0]
                    array_len = struct.unpack("<Q", file_obj.read(8))[0]
                    if sub_type in (0, 1, 7):
                        file_obj.seek(array_len * 1, os.SEEK_CUR)
                    elif sub_type in (2, 3):
                        file_obj.seek(array_len * 2, os.SEEK_CUR)
                    elif sub_type in (4, 5, 6):
                        file_obj.seek(array_len * 4, os.SEEK_CUR)
                    elif sub_type in (10, 11, 12):
                        file_obj.seek(array_len * 8, os.SEEK_CUR)
                    elif sub_type == 8:
                        for _ in range(array_len):
                            str_len = struct.unpack("<Q", file_obj.read(8))[0]
                            file_obj.seek(str_len, os.SEEK_CUR)
                    else:
                        for _ in range(array_len):
                            read_val(file_obj, sub_type)
                    return f"<Array of type {sub_type} length {array_len}>"
                elif v_type == 10: return struct.unpack("<Q", file_obj.read(8))[0]
                elif v_type == 11: return struct.unpack("<q", file_obj.read(8))[0]
                elif v_type == 12: return struct.unpack("<d", file_obj.read(8))[0]
                return None

            for _ in range(metadata_count):
                key = read_str(f)
                val_type = struct.unpack("<I", f.read(4))[0]
                val = read_val(f, val_type)
                kl = key.lower()
                if any(x in kl for x in ("block_count", "head_count", "embedding_length", "key_length", "value_length")):
                    metadata[key] = val
    except Exception:
        pass
    return metadata


ASCII_HEADER = """  ╦  ╦  ╔═╗ ╔╦╗ ╔═╗   ╦  ╔═╗ ╦ ╦ ╔╗╔ ╔═╗ ╦ ╦ ╔═╗ ╦═╗
  ║  ║  ╠═╣ ║║║ ╠═╣   ║  ╠═╣ ║ ║ ║║║ ║   ╠═╣ ║╣  ╠╦╝
  ╩═╝╩═╝╩ ╩ ╩ ╩ ╩ ╩   ╩═╝╩ ╩ ╚═╝ ╝╚╝ ╚═╝ ╩ ╩ ╚═╝ ╩╚═"""

DEFAULT_CONFIG = {
    "BIN": os.path.expanduser("~/llama.cpp/build/bin/llama-server"),
    "MODEL": os.path.expanduser("~/models/Qwen3.6-27B-UD-Q4_K_XL.gguf"),
    "MODEL_DRAFT": "",
    "HOST": "127.0.0.1",
    "PORT": "8080",
    "API_KEY": "",
    "ALIAS": "qwen3.6-27b-mtp",
    "NGL": "99",
    "CTX": "85000",
    "NP": "1",
    "THREADS": "-1",
    "THREADS_BATCH": "-1",
    "BATCH_SIZE": "2048",
    "UBATCH_SIZE": "512",
    "FLASH_ATTN": "on",
    "CACHE_K": "q8_0",
    "CACHE_V": "q8_0",
    "MLOCK": "off",
    "TEMP": "0.6",
    "TOP_P": "0.95",
    "TOP_K": "20",
    "MIN_P": "0.0",
    "PRESENCE_PENALTY": "0.0",
    "REPEAT_PENALTY": "1.0",
    "SPEC_TYPE": "draft-mtp",
    "SPEC_MAX": "3",
    "SPEC_MIN": "0.15",
    "SPEC_DRAFT_N_MIN": "0",
    "JINJA": "on",
    "REASONING": "auto",
    "REASONING_FORMAT": "auto",
    "REASONING_BUDGET": "-1",
    "METRICS": "off",
    "SEED": "-1",
    "CHAT_TEMPLATE": "",
    "CHAT_TEMPLATE_FILE": "",
    "TEMPLATE_KWARGS": '{"preserve_thinking": true}',
    "MMPROJ": "",
    "MMPROJ_OFFLOAD": "on",
    "SKIP_CHAT_PARSING": "off",
    "WEBUI": "on",
    "SPLIT_MODE": "none",
    "MAIN_GPU": "0",
    "PREDICT": "-1",
    "CTX_CHECKPOINTS": "",
    "CACHE_RAM": "",
    "CONT_BATCHING": "on",
    "DRAFT_MAX": "",
    "DRAFT_P_MIN": "",
    "IMAGE_MIN_TOKENS": "",
    "IMAGE_MAX_TOKENS": "",
    "TOOLS": "all",
    "TENSOR_SPLIT": "",
    "NUMA": "none",
    "NO_MMAP": "off",
    "TIMEOUT": "1800",
    "THREADS_HTTP": "-1",
    "CACHE_PROMPT": "on",
    "CACHE_REUSE": "256",
    "DRY_MULTIPLIER": "0.0",
    "DRY_BASE": "1.75",
    "DRY_ALLOWED_LENGTH": "2",
    "DRY_PENALTY_LAST_N": "-1",
    "XTC_PROBABILITY": "0.0",
    "XTC_THRESHOLD": "0.1",
    "SWA_FULL": "off",
    "GGML_CUDA_DISABLE_GRAPHS": "off",
    "GGML_CUDA_ENABLE_UNIFIED_MEMORY": "off",
    "EXTRA_FLAGS": "",
    "DISABLED_FIELDS": [
        "API_KEY",
        "MODEL_DRAFT",
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


PARAM_HELP = {
    "BIN": "[Flag: None] | [Default: ~/llama.cpp/build/bin/llama-server] | Path to llama-server binary executable. Ensure it is compiled with GPU support (CUDA/ROCm) for acceleration.",
    "MODEL": "[Flag: -m / --model] | [Default: ~/models/Qwen3.6-27B-UD-Q4_K_XL.gguf] | Path to the GGUF model file to load. Use Q4_K_M or Q4_K_XL quantizations for a great balance of model size and intelligence.",
    "HOST": "[Flag: --host] | [Default: 127.0.0.1] | IP address the server listens on. Use 0.0.0.0 to allow access from other devices on your local network.",
    "PORT": "[Flag: --port] | [Default: 8080] | TCP port number for the HTTP server. Change this if another service is already running on port 8080.",
    "API_KEY": "[Flag: --api-key] | [Default: None] | API key for authentication. Comma-separate for multiple keys. Leave empty to disable.",
    "ALIAS": "[Flag: -a / --alias] | [Default: None] | Model name alias returned in OpenAI-compatible API responses (e.g., /v1/models).",
    "WEBUI": "[Flag: --webui / --no-webui] | [Default: auto] | Enable and configure the built-in server web interface (auto/on/off). Auto enables it if web assets are compiled in.",
    "NGL": "[Flag: -ngl] | [Default: 99] | Number of layers offloaded to GPU. Use 99 for full offload. Lower values split execution between CPU and GPU.",
    "CTX": "[Flag: -c] | [Default: 32768] | Context window size in tokens. Qwen3.6 supports up to 262K natively. Note that larger contexts require significantly more VRAM.",
    "NP": "[Flag: -np] | [Default: 1] | Number of parallel request slots (concurrent users). Increasing this multiplies the KV Cache VRAM requirements.",
    "THREADS": "[Flag: -t] | [Default: -1 (auto)] | CPU threads for generation. Set to physical CPU core count for optimal performance.",
    "THREADS_BATCH": "[Flag: --threads-batch] | [Default: 8] | CPU threads for batch prompt evaluation during prefill. Set to physical core count.",
    "BATCH_SIZE": "[Flag: -b] | [Default: 2048] | Logical batch size for prompt evaluation. Larger values speed up prefill/prompt ingestion at the cost of VRAM.",
    "UBATCH_SIZE": "[Flag: -ub] | [Default: 512] | Physical GPU batch size. Smaller values reduce VRAM usage; larger values speed up processing. Max is the logical Batch Size.",
    "FLASH_ATTN": "[Flag: --flash-attn] | [Default: on] | Flash Attention (on/off/auto). Highly recommended to keep 'on' to drastically reduce KV cache memory usage.",
    "CACHE_K": "[Flag: --cache-type-k] | [Default: q8_0] | Quantization for KV cache keys. Format q8_0 halves memory compared to f16 with virtually no quality loss.",
    "CACHE_V": "[Flag: --cache-type-v] | [Default: q8_0] | Quantization for KV cache values. Format q8_0 is recommended. Use bf16 if model outputs become garbled.",
    "MLOCK": "[Flag: --mlock] | [Default: off] | Lock model in RAM to prevent OS paging/swapping. Requires sufficient physical memory to fit the model.",
    "SPLIT_MODE": "[Flag: -sm] | [Default: none] | How to split model across multiple GPUs (none/row/layer). Use 'none' for a single GPU.",
    "MAIN_GPU": "[Flag: -mg] | [Default: 0] | The primary GPU index (0, 1, etc.) used for coordinating tensor operations in multi-GPU setups.",
    "TENSOR_SPLIT": "[Flag: -ts] | [Default: None] | Fraction of model to allocate to each GPU (comma-separated, e.g., 3,1). Leave empty for automatic splitting.",
    "NUMA": "[Flag: --numa] | [Default: none] | NUMA optimization style (none/distribute/isolate/numactl). Recommended to leave as 'none' for single-socket systems.",
    "CTX_CHECKPOINTS": "[Flag: --ctx-checkpoints] | [Default: None] | Number of context checkpoints to store. Saves VRAM during long-context processing.",
    "CACHE_RAM": "[Flag: --cache-ram] | [Default: None] | GPU memory (in MB) allocated to the KV cache structure. Helps reserve VRAM safely.",
    "CONT_BATCHING": "[Flag: --cont-batching] | [Default: on] | Enable continuous batching of multiple request sequences. Highly recommended when slots (NP) > 1.",
    "CACHE_PROMPT": "[Flag: --cache-prompt] | [Default: on] | Enable prompt caching to reuse past context in multi-turn dialogues. Greatly speeds up chat history responses.",
    "CACHE_REUSE": "[Flag: --cache-reuse] | [Default: 256] | Minimum number of prompt tokens to reuse from cache. Prevents small, wasteful cache lookup updates.",
    "NO_MMAP": "[Flag: --no-mmap] | [Default: off] | Disable memory-mapping (mmap). Loads all weights directly into RAM to prevent disk swapping stutter.",
    "TIMEOUT": "[Flag: --timeout] | [Default: 1800] | Server connection timeout in seconds. Prevents client disconnects during long generations or heavy queues.",
    "THREADS_HTTP": "[Flag: --threads-http] | [Default: -1] | Number of worker threads for processing incoming HTTP requests. Set to -1 for automatic configuration.",
    "TEMP": "[Flag: --temp] | [Default: 0.6] | Sampling temperature. Qwen3.6 recommendations: 0.6 for coding/thinking, 0.7 for non-thinking, 1.0 for general reasoning.",
    "TOP_P": "[Flag: --top-p] | [Default: 0.95] | Nucleus sampling threshold. Qwen3.6 recommendations: 0.95 for thinking mode, 0.8 for non-thinking.",
    "TOP_K": "[Flag: --top-k] | [Default: 20] | Only sample from the top K tokens. Recommended value for Qwen3.6 is 20.",
    "MIN_P": "[Flag: --min-p] | [Default: 0.0] | Minimum probability threshold relative to the top token. Filter out low-probability noise (e.g., set to 0.05).",
    "PRESENCE_PENALTY": "[Flag: --presence-penalty] | [Default: 0.0] | Penalize tokens based on presence in text. Qwen3.6: 0.0 for thinking mode, 1.5 for non-thinking.",
    "REPEAT_PENALTY": "[Flag: --repeat-penalty] | [Default: 1.0] | Penalize repeated tokens (1.0 = no penalty). Keep at 1.0 for Qwen3.6 and prefer the DRY sampler instead.",
    "PREDICT": "[Flag: --predict] | [Default: -1] | Maximum number of tokens to generate per request. Set to -1 for unlimited generation.",
    "SEED": "[Flag: -s] | [Default: -1 (random)] | RNG seed for reproducible output. Set to a positive integer to get deterministic generation.",
    "DRY_MULTIPLIER": "[Flag: --dry-multiplier] | [Default: 0.8] | DRY sampler multiplier. 0.8 is optimal for Qwen3.6 to prevent repetitive phrasing loops.",
    "DRY_BASE": "[Flag: --dry-base] | [Default: 1.75] | DRY sampler base penalty exponent. Determines the scaling curve of repetition penalization.",
    "DRY_ALLOWED_LENGTH": "[Flag: --dry-allowed-length] | [Default: 2] | Number of matching tokens allowed in sequence before DRY penalty starts applying.",
    "DRY_PENALTY_LAST_N": "[Flag: --dry-penalty-last-n] | [Default: -1] | Limit DRY search window to the last N tokens. Set to -1 to scan the entire context.",
    "XTC_PROBABILITY": "[Flag: --xtc-probability] | [Default: 0.5] | XTC sampler probability. Balances logical flow and vocabulary diversity (e.g., 0.5).",
    "XTC_THRESHOLD": "[Flag: --xtc-threshold] | [Default: 0.1] | XTC sampler minimum probability threshold. Excludes predictable choices below threshold.",
    "MODEL_DRAFT": "[Flag: -md / --model-draft] | [Default: None] | Path to the speculative decoding draft model (GGUF file). Used to accelerate generation via speculative draft-model decoding.",
    "SPEC_TYPE": "[Flag: --spec-type] | [Default: draft-mtp] | Speculative decoding method. Use 'draft-mtp' for Qwen3.6-MTP native multi-token prediction.",
    "SPEC_MAX": "[Flag: --spec-draft-n-max] | [Default: 3] | Maximum draft tokens to speculate per step. 2 to 3 is optimal for Qwen3.6-MTP models.",
    "SPEC_MIN": "[Flag: --spec-draft-p-min] | [Default: 0.05] | Minimum acceptance probability for draft tokens. Lower values speculate more aggressively.",
    "SPEC_DRAFT_N_MIN": "[Flag: --spec-draft-n-min] | [Default: 0] | Minimum draft tokens evaluated per step. Values above 0 prevent short, inefficient speculations.",
    "DRAFT_MAX": "[Flag: --draft-max] | [Default: None] | Maximum speculative draft tokens to generate per step. Leave empty to use model default.",
    "DRAFT_P_MIN": "[Flag: --draft-p-min] | [Default: None] | Minimum acceptance probability for standard draft model tokens.",
    "REASONING": "[Flag: --reasoning] | [Default: auto] | Qwen3.6 thinking mode: on (always think), off (never think), auto (model decides based on query).",
    "REASONING_FORMAT": "[Flag: --reasoning-format] | [Default: auto] | Output format for <think> blocks (auto/deepseek/none). Use 'deepseek' for OpenAI-compatible clients.",
    "REASONING_BUDGET": "[Flag: --reasoning-budget] | [Default: -1] | Maximum thinking tokens allowed. Set to -1 for unlimited, 0 to disable reasoning, or N for limit.",
    "TOOLS": "[Flag: --tools] | [Default: all] | Enable model tool-calling feature (all/none/comma-separated-list).",
    "JINJA": "[Flag: --jinja / --no-jinja] | [Default: on] | Enable Jinja2 chat template rendering. Required for tool calling and reasoning tags formatting.",
    "METRICS": "[Flag: --metrics] | [Default: off] | Enable Prometheus metrics endpoint at /metrics. Useful for grafana dashboards and profiling.",
    "MMPROJ": "[Flag: --mmproj] | [Default: None] | Path to multimodal projector file (needed only for vision-language GGUF models).",
    "MMPROJ_OFFLOAD": "[Flag: --no-mmproj-offload (if off)] | [Default: on] | Offload vision projector math to GPU. Keeps visual prefill latency low.",
    "IMAGE_MIN_TOKENS": "[Flag: --image-min-tokens] | [Default: None] | Minimum KV Cache token budget reserved for each input image.",
    "IMAGE_MAX_TOKENS": "[Flag: --image-max-tokens] | [Default: None] | Maximum KV Cache token budget allowed for each input image.",
    "CHAT_TEMPLATE": "[Flag: --chat-template] | [Default: None] | Force a built-in static chat template format (e.g., chatml, llama3). Ignored if Jinja is enabled.",
    "CHAT_TEMPLATE_FILE": "[Flag: --chat-template-file] | [Default: None] | Path to a custom Jinja chat template file, overriding the model's embedded template.",
    "TEMPLATE_KWARGS": "[Flag: --chat-template-kwargs] | [Default: {\"preserve_thinking\":true}] | JSON arguments for chat templates. Crucial for Qwen3.6/DeepSeek to keep thinking tags.",
    "SKIP_CHAT_PARSING": "[Flag: --skip-chat-parsing] | [Default: off] | Disable internal parser. If enabled, reasoning/thinking and tool calls are dumped directly into content.",
    "SWA_FULL": "[Flag: --swa-full] | [Default: off] | Use full-size Sliding Window Attention (SWA) cache. Prevents context invalidation and reprocessing on models using SWA (like Qwen 3.6).",
    "GGML_CUDA_DISABLE_GRAPHS": "[Flag: GGML_CUDA_DISABLE_GRAPHS=1] | [Default: on] | Disable CUDA Graphs (on/off). Highly recommended to keep 'on' when using quantized KV Cache (q8_0) under Flash Attention to prevent GPU illegal memory access crashes.",
    "GGML_CUDA_ENABLE_UNIFIED_MEMORY": "[Flag: GGML_CUDA_ENABLE_UNIFIED_MEMORY=1] | [Default: off] | Enable Unified Memory (on/off). Allows the GPU to fall back to system RAM when VRAM is exhausted. Prevents Out-Of-Memory crashes but significantly degrades performance. Keep 'off' for strict VRAM bounds.",
    "EXTRA_FLAGS": "[Flag: None] | [Default: None] | Additional custom CLI arguments to append directly to the command (e.g., --verbose --grp-attn-n 4).",
}

SELECT_OPTIONS = {
    "FLASH_ATTN": [("on", "on"), ("off", "off"), ("auto", "auto")],
    "MLOCK": [("on", "on"), ("off", "off")],
    "METRICS": [("on", "on"), ("off", "off")],
    "MMPROJ_OFFLOAD": [("on", "on"), ("off", "off")],
    "SKIP_CHAT_PARSING": [("on", "on"), ("off", "off")],
    "CONT_BATCHING": [("on", "on"), ("off", "off")],
    "JINJA": [("on", "on"), ("off", "off")],
    "WEBUI": [("auto", "auto"), ("on", "on"), ("off", "off")],
    "SPLIT_MODE": [("none", "none"), ("row", "row"), ("layer", "layer")],
    "REASONING": [("auto", "auto"), ("on", "on"), ("off", "off")],
    "REASONING_FORMAT": [("auto", "auto"), ("deepseek", "deepseek"), ("none", "none")],
    "NUMA": [("none", "none"), ("distribute", "distribute"), ("isolate", "isolate"), ("numactl", "numactl")],
    "NO_MMAP": [("off", "off"), ("on", "on")],
    "CACHE_PROMPT": [("on", "on"), ("off", "off")],
    "SWA_FULL": [("off", "off"), ("on", "on")],
    "GGML_CUDA_DISABLE_GRAPHS": [("on", "on"), ("off", "off")],
    "GGML_CUDA_ENABLE_UNIFIED_MEMORY": [("off", "off"), ("on", "on")],
}

def normalize_select_value(key: str, value: str) -> str:
    """Normalize boolean, number, and case-variations for Select widgets."""
    if key not in SELECT_OPTIONS:
        return str(value)
    opts = SELECT_OPTIONS[key]
    valid_vals = [o[1] for o in opts]
    val_str = str(value).strip().lower()
    
    if value in valid_vals:
        return value
        
    if val_str in ("true", "1", "yes", "on") and "on" in valid_vals:
        return "on"
    elif val_str in ("false", "0", "no", "off") and "off" in valid_vals:
        return "off"
    elif val_str in ("auto",) and "auto" in valid_vals:
        return "auto"
    elif val_str in ("none",) and "none" in valid_vals:
        return "none"
        
    return valid_vals[0]

FLAG_MAPPING = {
    "MODEL": "-m",
    "MODEL_DRAFT": "-md",
    "HOST": "--host",
    "PORT": "--port",
    "NGL": "-ngl",
    "CTX": "-c",
    "NP": "-np",
    "THREADS": "-t",
    "BATCH_SIZE": "-b",
    "UBATCH_SIZE": "-ub",
    "TEMP": "--temp",
    "TOP_P": "--top-p",
    "TOP_K": "--top-k",
    "MIN_P": "--min-p",
    "PRESENCE_PENALTY": "--presence-penalty",
    "REPEAT_PENALTY": "--repeat-penalty",
    "CACHE_K": "--cache-type-k",
    "CACHE_V": "--cache-type-v",
    "SPEC_TYPE": "--spec-type",
    "SPEC_MAX": "--spec-draft-n-max",
    "SPEC_MIN": "--spec-draft-p-min",
    "SPEC_DRAFT_N_MIN": "--spec-draft-n-min",
    "SEED": "-s",
    "REASONING": "--reasoning",
    "REASONING_FORMAT": "--reasoning-format",
    "REASONING_BUDGET": "--reasoning-budget",
    "CHAT_TEMPLATE": "--chat-template",
    "CHAT_TEMPLATE_FILE": "--chat-template-file",
    "TEMPLATE_KWARGS": "--chat-template-kwargs",
    "MMPROJ": "--mmproj",
    "CTX_CHECKPOINTS": "--ctx-checkpoints",
    "CACHE_RAM": "--cache-ram",
    "THREADS_BATCH": "--threads-batch",
    "SPLIT_MODE": "--split-mode",
    "MAIN_GPU": "--main-gpu",
    "PREDICT": "--predict",
    "DRAFT_MAX": "--draft-max",
    "DRAFT_P_MIN": "--draft-p-min",
    "IMAGE_MIN_TOKENS": "--image-min-tokens",
    "IMAGE_MAX_TOKENS": "--image-max-tokens",
    "TOOLS": "--tools",
    "TENSOR_SPLIT": "--tensor-split",
    "TIMEOUT": "--timeout",
    "THREADS_HTTP": "--threads-http",
    "CACHE_REUSE": "--cache-reuse",
    "DRY_MULTIPLIER": "--dry-multiplier",
    "DRY_BASE": "--dry-base",
    "DRY_ALLOWED_LENGTH": "--dry-allowed-length",
    "DRY_PENALTY_LAST_N": "--dry-penalty-last-n",
    "XTC_PROBABILITY": "--xtc-probability",
    "XTC_THRESHOLD": "--xtc-threshold",
}

def build_server_args(params: dict, enabled_fields: dict = None) -> list[str]:
    """Construct argument list for llama-server from parameters."""
    cmd = []
    
    # 1. Map standard parameters to standard flags
    for key, flag in FLAG_MAPPING.items():
        if key in params:
            if enabled_fields and not enabled_fields.get(key, True):
                continue
            val = str(params[key]).strip()
            if val:
                cmd.extend([flag, val])
                
    # 2. Build non-standard/special flags
    def is_active(k):
        if enabled_fields and not enabled_fields.get(k, True):
            return False
        return k in params

    if is_active("FLASH_ATTN"):
        cmd.extend(["--flash-attn", params["FLASH_ATTN"]])
    if is_active("JINJA"):
        if params["JINJA"].lower() in ("on", "1", "true", "yes"):
            cmd.append("--jinja")
        else:
            cmd.append("--no-jinja")
    if is_active("MLOCK"):
        if params["MLOCK"].lower() in ("on", "1", "true", "yes"):
            cmd.append("--mlock")
    if is_active("SWA_FULL"):
        if params["SWA_FULL"].lower() in ("on", "1", "true", "yes"):
            cmd.append("--swa-full")
    if is_active("METRICS"):
        if params["METRICS"].lower() in ("on", "1", "true", "yes"):
            cmd.append("--metrics")
    if is_active("WEBUI"):
        if params["WEBUI"].lower() in ("on", "1", "true", "yes", "auto"):
            cmd.append("--webui")
        else:
            cmd.append("--no-webui")
    if is_active("CONT_BATCHING"):
        if params["CONT_BATCHING"].lower() in ("on", "1", "true", "yes"):
            cmd.append("--cont-batching")
        else:
            cmd.append("--no-cont-batching")
    if is_active("API_KEY") and params["API_KEY"].strip():
        cmd.extend(["--api-key", params["API_KEY"].strip()])
    if is_active("ALIAS") and params["ALIAS"].strip():
        cmd.extend(["-a", params["ALIAS"].strip()])
    if is_active("MMPROJ_OFFLOAD"):
        if params["MMPROJ_OFFLOAD"].lower() in ("off", "0", "false", "no"):
            cmd.append("--no-mmproj-offload")
    if is_active("SKIP_CHAT_PARSING"):
        if params["SKIP_CHAT_PARSING"].lower() in ("on", "1", "true", "yes"):
            cmd.append("--skip-chat-parsing")
    if is_active("NUMA"):
        if params["NUMA"] != "none":
            cmd.extend(["--numa", params["NUMA"]])
    if is_active("NO_MMAP"):
        if params["NO_MMAP"] == "on":
            cmd.append("--no-mmap")
    if is_active("CACHE_PROMPT"):
        if params["CACHE_PROMPT"] == "on":
            cmd.append("--cache-prompt")
        else:
            cmd.append("--no-cache-prompt")
    if is_active("EXTRA_FLAGS") and params["EXTRA_FLAGS"].strip():
        cmd.extend(shlex.split(params["EXTRA_FLAGS"].strip()))
        
    return cmd

class ParameterField(Horizontal):
    enabled = reactive(True)

    def __init__(self, key: str, label: str, value: str, is_mandatory: bool = False, is_enabled: bool = True):
        super().__init__()
        self.key = key
        self.label_text = label
        self.true_value = value
        
        if self.key in SELECT_OPTIONS:
            self.true_value = normalize_select_value(self.key, value)

        self.initial_value = self.true_value
        if self.key in ("MODEL", "BIN", "MMPROJ", "CHAT_TEMPLATE_FILE", "MODEL_DRAFT"):
            self.initial_value = truncate_path(self.true_value)
        self.is_mandatory = is_mandatory
        self._initial_enabled = is_enabled

    def on_descendant_focus(self, event) -> None:
        # Restore full absolute path when focused
        inp = self.get_input_widget()
        if inp is not None and hasattr(self, "true_value"):
            if not isinstance(inp, Select):
                inp.value = self.true_value

    def on_descendant_blur(self, event) -> None:
        # Save and truncate path when blurred
        inp = self.get_input_widget()
        if inp is not None:
            if not isinstance(inp, Select):
                val_str = inp.value
                if not (self.key in ("MODEL", "BIN", "MMPROJ", "CHAT_TEMPLATE_FILE", "MODEL_DRAFT") and val_str.startswith(".../")):
                    self.true_value = val_str
            if self.key in ("MODEL", "BIN", "MMPROJ", "CHAT_TEMPLATE_FILE", "MODEL_DRAFT"):
                inp.value = truncate_path(self.true_value)

    def get_input_widget(self):
        widget = getattr(self, "_input_widget", None)
        if widget is None:
            try:
                if self.key in SELECT_OPTIONS:
                    widget = self.query_one(Select)
                else:
                    widget = self.query_one(Input)
                self._input_widget = widget
            except Exception:
                widget = None
        return widget

    def get_chk_widget(self):
        widget = getattr(self, "_chk_widget", None)
        if widget is None:
            try:
                widget = self.query_one(f"#chk-{self.key}", Label)
                self._chk_widget = widget
            except Exception:
                widget = None
        return widget

    def get_label_widget(self):
        widget = getattr(self, "_label_widget", None)
        if widget is None:
            try:
                widget = self.query_one(".field-label", Label)
                self._label_widget = widget
            except Exception:
                widget = None
        return widget

    def get_info_widget(self):
        widget = getattr(self, "_info_widget", None)
        if widget is None:
            try:
                widget = self.query_one(".info-icon", Label)
                self._info_widget = widget
            except Exception:
                widget = None
        return widget

    def get_btn_widget(self):
        widget = getattr(self, "_btn_widget", None)
        if widget is None:
            try:
                widget = self.query_one(".browse-btn", Button)
                self._btn_widget = widget
            except Exception:
                widget = None
        return widget

    @property
    def value(self) -> str:
        return getattr(self, "true_value", "")

    @value.setter
    def value(self, val: str) -> None:
        if self.key in SELECT_OPTIONS:
            val = normalize_select_value(self.key, val)

        self.true_value = val
        inp = self.get_input_widget()
        if inp is not None:
            if inp.has_focus:
                inp.value = val
            else:
                if self.key in ("MODEL", "BIN", "MMPROJ", "CHAT_TEMPLATE_FILE", "MODEL_DRAFT"):
                    inp.value = truncate_path(val)
                else:
                    inp.value = val

    def compose(self) -> ComposeResult:
        if not self.is_mandatory:
            yield Label(" ✓ ", classes="custom-checkbox", id=f"chk-{self.key}")
        else:
            yield Label("   ", classes="spacer-checkbox")
        yield Label(self.label_text, classes="field-label")
        if self.key in PARAM_HELP:
            info = Label(" ⓘ ", classes="info-icon")
            info.tooltip = PARAM_HELP[self.key]
            yield info
        
        if self.key in SELECT_OPTIONS:
            opts = SELECT_OPTIONS[self.key]
            val = self.initial_value if any(o[1] == self.initial_value for o in opts) else opts[0][1]
            yield Select(opts, value=val, id=self.key, allow_blank=False)
        else:
            yield Input(value=str(self.initial_value), id=self.key)
            
        if self.key == "MODEL":
            yield Button("📂", id="btn-browse-model", classes="browse-btn")
        elif self.key == "BIN":
            yield Button("📂", id="btn-browse-bin", classes="browse-btn")
        elif self.key == "MMPROJ":
            yield Button("📂", id="btn-browse-mmproj", classes="browse-btn")
        elif self.key == "CHAT_TEMPLATE_FILE":
            yield Button("📂", id="btn-browse-template", classes="browse-btn")
        elif self.key == "MODEL_DRAFT":
            yield Button("📂", id="btn-browse-model-draft", classes="browse-btn")

    def watch_enabled(self, enabled: bool) -> None:
        self.update_visuals()
        if hasattr(self, "app") and self.app and hasattr(self.app, "update_dashboard"):
            self.app.update_dashboard()

    def update_visuals(self) -> None:
        if not self.is_mounted:
            return
        if not self.is_mandatory:
            # Custom Checkbox Label
            chk = self.get_chk_widget()
            if chk is not None:
                chk.update(" ✓ " if self.enabled else " ✗ ")
                if self.enabled:
                    chk.remove_class("dimmed")
                else:
                    chk.add_class("dimmed")
            
            # Field Label
            lbl = self.get_label_widget()
            if lbl is not None:
                 
                if self.enabled:
                    lbl.remove_class("dimmed")
                else:
                    lbl.add_class("dimmed")
                    
            # Info Icon
            info = self.get_info_widget()
            if info is not None:
                 
                if self.enabled:
                    info.remove_class("dimmed")
                else:
                    info.add_class("dimmed")

            # Self (ParameterField)
            if self.enabled:
                self.remove_class("dimmed")
            else:
                self.add_class("dimmed")
                
            # Input Widget
            inp = self.get_input_widget()
            if inp is not None:
                inp.disabled = not self.enabled
                if self.enabled:
                    inp.remove_class("dimmed")
                else:
                    inp.add_class("dimmed")
            
            # Browse Button Widget
            btn = self.get_btn_widget()
            if btn is not None:
                btn.disabled = not self.enabled
                if self.enabled:
                    btn.remove_class("dimmed")
                else:
                    btn.add_class("dimmed")

    def on_mount(self) -> None:
        self.enabled = self._initial_enabled
        self.call_after_refresh(self.update_visuals)

    def on_click(self, event) -> None:
        if not self.is_mandatory and hasattr(event, "control") and event.control:
            # Allow toggling either by clicking the checkbox itself or the text label next to it
            if event.control.id == f"chk-{self.key}" or (hasattr(event.control, "classes") and "field-label" in event.control.classes):
                self.enabled = not self.enabled

    @on(Select.Changed)
    def on_select_changed(self, event: Select.Changed) -> None:
        self.true_value = str(event.value)
        logging.debug(f"[ParameterField select_changed] key={self.key}, val={event.value}")
        if hasattr(self, "app") and self.app and hasattr(self.app, "update_dashboard"):
            self.app.update_dashboard()

    @on(Input.Changed)
    def on_input_changed(self, event: Input.Changed) -> None:
        val_str = str(event.value)
        if self.key in ("MODEL", "BIN", "MMPROJ", "CHAT_TEMPLATE_FILE", "MODEL_DRAFT") and val_str.startswith(".../"):
            return  # Ignore visual path mask truncation updates to preserve the full absolute path
        self.true_value = val_str
        logging.debug(f"[ParameterField input_changed] key={self.key}, val={event.value}")
        if hasattr(self, "app") and self.app:
            if hasattr(self.app, "debounce_update_dashboard"):
                self.app.debounce_update_dashboard()
            elif hasattr(self.app, "update_dashboard"):
                self.app.update_dashboard()




from typing import TypeVar, Generic
T = TypeVar('T')

class BaseModal(ModalScreen[T], Generic[T]):
    """Base modal class with shared styling for all modals."""
    DEFAULT_CSS = """
    """

class FileBrowserModal(BaseModal[str]):
    """Modal file browser for selecting model files."""

    BINDINGS = [("escape", "cancel", "Cancel")]

    DEFAULT_CSS = """
    #file-browser-container {
        width: 90%;
        max-width: 90;
        height: 30;
        background: #282828;
        border: thick #d97757;
        padding: 1 2;
        overflow: hidden;
    }
    #fb-title {
        text-align: center;
        color: #d97757;
        text-style: bold;
        background: #282828;
        width: 100%;
        height: 1;
        margin-bottom: 1;
        overflow: hidden;
    }
    #fb-path {
        color: #d97757;
        background: #282828;
        width: 100%;
        height: 1;
        margin-bottom: 1;
        overflow: hidden;
    }
    #fb-file-list {
        height: 1fr;
        background: #1d2021;
        padding: 0 1;
        width: 100%;
        overflow-x: hidden;
    }
    .fb-entry {
        width: 100%;
        height: 1;
        background: #1d2021;
        color: #ebdbb2;
        padding: 0 1;
        overflow: hidden;
    }
    .fb-entry:hover {
        background: #3c3836;
    }
    .fb-entry-dir {
        color: #6a9bcc;
    }
    .fb-entry-file {
        color: #788c5d;
    }
    .fb-entry-parent {
        color: #d97757;
        text-style: bold;
    }
    #fb-buttons {
        height: auto;
        align: center middle;
        margin-top: 1;
        background: #282828;
    }
    #fb-buttons Button {
        margin: 0 1;
        min-width: 12;
        height: 1;
        min-height: 0;
        border: none;
    }
    #btn-fb-cancel {
        background: #665c54;
        color: #ebdbb2;
    }
    """

    def __init__(self, start_path: str = "", file_extensions: tuple | None = (".gguf", ".bin", ".ggml")):
        super().__init__()
        self.file_extensions = file_extensions
        if start_path and os.path.exists(start_path):
            if os.path.isfile(start_path):
                self.current_dir = Path(start_path).parent
            else:
                self.current_dir = Path(start_path)
        else:
            self.current_dir = Path.home()

    def get_display_path(self) -> str:
        path_str = str(self.current_dir)
        term_width = self.app.size.width if hasattr(self, "app") and self.app else 80
        usable_width = max(30, int(term_width * 0.85) - 8)
        if len(path_str) > usable_width:
            half = (usable_width - 5) // 2
            return path_str[:half] + "..." + path_str[-half:]
        return path_str

    def compose(self) -> ComposeResult:
        with Container(id="file-browser-container"):
            title = "📂 Select File" if self.file_extensions is None else "📂 Select Model File"
            yield Static(title, id="fb-title")
            yield Static(self.get_display_path(), id="fb-path")
            with VerticalScroll(id="fb-file-list"):
                yield from self._build_entries()
            with Horizontal(id="fb-buttons"):
                yield Button("Cancel", id="btn-fb-cancel", classes="modal-btn")

    def _build_entries(self):
        """Build file/dir labels for current directory."""
        term_width = self.app.size.width if hasattr(self, "app") and self.app else 80
        usable_w = max(40, int(term_width * 0.8) - 10)

        # Parent dir entry
        try:
            if self.current_dir != self.current_dir.parent:
                lbl = Label("📁 ..", classes="fb-entry fb-entry-parent")
                lbl.fb_path = str(self.current_dir.parent)
                lbl.fb_type = "parent"
                yield lbl
        except Exception:
            pass

        try:
            items = sorted(self.current_dir.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
        except PermissionError:
            yield Label("⚠ Permission denied", classes="fb-entry")
            return
        except Exception as e:
            yield Label(f"⚠ Error: {e}", classes="fb-entry")
            return

        for item in items:
            try:
                if item.is_dir():
                    dirname = item.name
                    max_dir_len = usable_w - 4
                    if len(dirname) > max_dir_len:
                        half = (max_dir_len - 3) // 2
                        display_name = dirname[:half] + "..." + dirname[-half:]
                    else:
                        display_name = dirname
                    lbl = Label(f"📁 {display_name}", classes="fb-entry fb-entry-dir")
                    lbl.fb_path = str(item)
                    lbl.fb_type = "dir"
                    yield lbl
                elif item.is_file():
                    if self.file_extensions is not None and item.suffix.lower() not in self.file_extensions:
                        continue
                    
                    try:
                        size_mb = item.stat().st_size / (1024 * 1024)
                        size_str = f"{size_mb:.0f}MB" if size_mb < 1024 else f"{size_mb/1024:.1f}GB"
                    except Exception:
                        size_str = "? MB"
                    
                    filename = item.name
                    max_file_len = usable_w - len(size_str) - 8
                    if len(filename) > max_file_len:
                        ext = item.suffix
                        base = item.stem
                        base_max = max_file_len - len(ext) - 3
                        if base_max > 6:
                            half = base_max // 2
                            display_name = base[:half] + "..." + base[-half:] + ext
                        else:
                            half = (max_file_len - 3) // 2
                            display_name = filename[:half] + "..." + filename[-half:]
                    else:
                        display_name = filename

                    lbl = Label(f"📄 {display_name}  ({size_str})", classes="fb-entry fb-entry-file")
                    lbl.fb_path = str(item)
                    lbl.fb_type = "file"
                    yield lbl
            except Exception as e:
                logging.warning(f"Error rendering item {item} in FileBrowserModal: {e}")
                continue

    def _refresh_list(self):
        """Refresh the file list for current directory."""
        self.query_one("#fb-path", Static).update(self.get_display_path())
        file_list = self.query_one("#fb-file-list", VerticalScroll)
        file_list.remove_children()
        file_list.mount(*list(self._build_entries()))

    def on_click(self, event) -> None:
        """Handle clicks on file/dir entries."""
        widget = event.control if hasattr(event, 'control') else None
        if widget is None or not isinstance(widget, Label):
            return
        if not hasattr(widget, "fb_type"):
            return

        if widget.fb_type == "parent":
            self.current_dir = self.current_dir.parent
            self._refresh_list()
        elif widget.fb_type == "dir":
            self.current_dir = Path(widget.fb_path)
            self._refresh_list()
        elif widget.fb_type == "file":
            self.dismiss(widget.fb_path)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-fb-cancel":
            self.dismiss("")

    def action_cancel(self) -> None:
        self.dismiss("")


class SaveProfileModal(BaseModal[str]):
    """Modal input dialog for saving a profile as a name."""

    BINDINGS = [("escape", "cancel", "Cancel")]

    DEFAULT_CSS = """
    #save-profile-container {
        width: 90%;
        max-width: 55;
        height: 19;
        background: #282828;
        border: thick #d97757;
        padding: 1 2;
        align: center middle;
    }
    #sp-title {
        color: #d97757;
        text-style: bold;
        text-align: center;
        width: 100%;
        margin-bottom: 1;
        background: #282828;
    }
    #sp-select-label {
        color: #ebdbb2;
        width: 100%;
        margin-bottom: 0;
        background: #282828;
    }
    #sp-label {
        color: #ebdbb2;
        width: 100%;
        margin-top: 1;
        margin-bottom: 0;
        background: #282828;
    }
    #sp-select {
        width: 100%;
        height: 1;
        border: none;
        background: transparent;
        margin-bottom: 1;
    }
    #sp-select SelectCurrent {
        width: 100%;
        background: #3c3836;
        color: #ebdbb2;
        border: none;
        height: 1;
        padding: 0 1;
    }
    #sp-select SelectCurrent Static#label {
        width: 1fr;
    }
    #sp-select > SelectOverlay {
        width: 1fr;
    }
    #sp-input {
        width: 100%;
        background: #3c3836;
        border: none;
        color: #ebdbb2;
        padding: 0 1;
        height: 1;
        margin-bottom: 1;
    }
    #sp-input:focus {
        background: #d97757;
        color: #282828;
    }
    #sp-buttons {
        align: center middle;
        height: 1;
        margin-top: 1;
        background: #282828;
    }
    #sp-buttons Button {
        margin: 0 1;
        min-width: 12;
        height: 1;
        min-height: 0;
        border: none;
    }
    #btn-sp-cancel {
        background: #665c54;
        color: #ebdbb2;
    }
    #btn-sp-save {
        background: #d97757;
        color: #282828;
        text-style: bold;
    }
    """

    def __init__(self, profiles_dir: str, current_name: str = "default"):
        super().__init__()
        self.profiles_dir = profiles_dir
        self.current_name = current_name

    def _get_profiles(self) -> list[tuple[str, str]]:
        opts = [("-- Choose Existing Profile (Optional) --", "")]
        try:
            files = sorted(os.listdir(self.profiles_dir))
            for f in files:
                if f.endswith(".json"):
                    name = f[:-5]
                    opts.append((f"📄 {name}", name))
        except Exception:
            pass
        return opts

    def compose(self) -> ComposeResult:
        opts = self._get_profiles()
        with Container(id="save-profile-container"):
            yield Static("💾 Save Profile As", id="sp-title")
            yield Label("Choose existing profile to overwrite:", id="sp-select-label")
            yield Select(opts, value="", id="sp-select", allow_blank=False)
            yield Label("Or enter new profile name:", id="sp-label")
            yield Input(value=self.current_name, id="sp-input")
            with Horizontal(id="sp-buttons"):
                yield Button("Cancel", id="btn-sp-cancel", classes="modal-btn")
                yield Button("Save", id="btn-sp-save", classes="modal-btn")

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.value:
            self.query_one("#sp-input", Input).value = str(event.value)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-sp-cancel":
            self.dismiss("")
        elif event.button.id == "btn-sp-save":
            self.submit()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.submit()

    def submit(self) -> None:
        val = self.query_one("#sp-input", Input).value.strip()
        safe_val = "".join(c for c in val if c.isalnum() or c in ("-", "_")).strip()
        if safe_val:
            self.dismiss(safe_val)
        else:
            self.dismiss("")

    def action_cancel(self) -> None:
        self.dismiss("")


class ConfirmDeleteModal(BaseModal[bool]):
    """Modal dialog to confirm deleting a profile."""
    
    BINDINGS = [
        ("escape", "cancel", "Cancel"),
        ("y", "confirm", "Confirm"),
        ("n", "cancel", "Cancel"),
    ]

    DEFAULT_CSS = """
    #confirm-container {
        width: 50;
        height: 11;
        background: #282828;
        border: thick #d97757;
        padding: 1 2;
        align: center middle;
    }
    #confirm-title {
        color: #d97757;
        text-style: bold;
        text-align: center;
        width: 100%;
        margin-bottom: 1;
        background: #282828;
    }
    #confirm-message {
        color: #ebdbb2;
        text-align: center;
        width: 100%;
        margin-bottom: 1;
        background: #282828;
    }
    #confirm-buttons {
        align: center middle;
        height: 1;
        background: #282828;
    }
    #confirm-buttons Button {
        margin: 0 1;
        min-width: 12;
        height: 1;
        min-height: 0;
        border: none;
    }
    #btn-confirm-yes {
        background: #d15d5d;
        color: #282828;
        text-style: bold;
    }
    #btn-confirm-no {
        background: #665c54;
        color: #ebdbb2;
    }
    """

    def __init__(self, profile_name: str):
        super().__init__()
        self.profile_name = profile_name

    def compose(self) -> ComposeResult:
        with Container(id="confirm-container"):
            yield Static("⚠ Confirm Delete", id="confirm-title")
            yield Static(f"Are you sure you want to delete profile '{self.profile_name}'?", id="confirm-message")
            with Horizontal(id="confirm-buttons"):
                yield Button("Yes, Delete", id="btn-confirm-yes", classes="modal-btn")
                yield Button("No", id="btn-confirm-no", classes="modal-btn")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-confirm-yes":
            self.dismiss(True)
        else:
            self.dismiss(False)

    def action_confirm(self) -> None:
        self.dismiss(True)

    def action_cancel(self) -> None:
        self.dismiss(False)


class LoadProfileModal(BaseModal[str]):
    """Modal dialog to list and select profiles."""

    BINDINGS = [("escape", "cancel", "Cancel")]

    DEFAULT_CSS = """
    #load-profile-container {
        width: 90%;
        max-width: 50;
        height: 19;
        background: #282828;
        border: thick #d97757;
        padding: 1 2;
        align: center middle;
    }
    #lp-title {
        color: #d97757;
        text-style: bold;
        text-align: center;
        width: 100%;
        margin-bottom: 1;
        background: #282828;
    }
    #lp-list {
        height: 1fr;
        background: #1d2021;
        padding: 0 1;
        width: 100%;
        margin-bottom: 1;
    }
    .lp-entry {
        width: 1fr;
        height: 1;
        background: transparent;
        color: #ebdbb2;
        padding: 0 1;
    }
    .lp-entry:hover {
        color: #6a9bcc;
        text-style: bold;
    }
    #lp-buttons {
        align: center middle;
        height: 1;
        background: #282828;
    }
    #lp-buttons Button {
        margin: 0 1;
        min-width: 12;
        height: 1;
        min-height: 0;
        border: none;
    }
    #btn-lp-cancel {
        background: #665c54;
        color: #ebdbb2;
    }
    """

    def __init__(self, profiles_dir: str):
        super().__init__()
        self.profiles_dir = profiles_dir

    def compose(self) -> ComposeResult:
        with Container(id="load-profile-container"):
            yield Static("📂 Load Profile", id="lp-title")
            with VerticalScroll(id="lp-list"):
                yield from self._build_entries()
            with Horizontal(id="lp-buttons"):
                yield Button("Cancel", id="btn-lp-cancel", classes="modal-btn")

    def _build_entries(self):
        try:
            files = sorted(os.listdir(self.profiles_dir))
            for f in files:
                if f.endswith(".json"):
                    name = f[:-5]
                    lbl = Label(f"📄 {name}", classes="lp-entry")
                    lbl.profile_name = name
                    yield lbl
        except Exception:
            yield Label("⚠ Failed to load profiles", classes="lp-entry")

    def on_click(self, event) -> None:
        widget = event.control if hasattr(event, 'control') else None
        if widget is None or not isinstance(widget, Label):
            return
        if hasattr(widget, "profile_name"):
            self.dismiss(widget.profile_name)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-lp-cancel":
            self.dismiss("")

    def action_cancel(self) -> None:
        self.dismiss("")


class DeleteProfileModal(BaseModal[None]):
    """Modal dialog to list and delete custom profiles."""

    BINDINGS = [("escape", "cancel", "Cancel")]

    DEFAULT_CSS = """
    #delete-profile-container {
        width: 90%;
        max-width: 50;
        height: 19;
        background: #282828;
        border: thick #d97757;
        padding: 1 2;
        align: center middle;
    }
    #dp-title {
        color: #d97757;
        text-style: bold;
        text-align: center;
        width: 100%;
        margin-bottom: 1;
        background: #282828;
    }
    #dp-list {
        height: 1fr;
        background: #1d2021;
        padding: 0 1;
        width: 100%;
        margin-bottom: 1;
    }
    .dp-entry {
        width: 100%;
        height: 1;
        background: transparent;
        color: #ebdbb2;
        padding: 0 1;
    }
    .dp-entry:hover {
        color: #d15d5d;
        text-style: bold;
    }
    .dp-entry.selected {
        background: #d15d5d;
        color: #1d2021;
        text-style: bold;
    }
    #dp-buttons {
        align: center middle;
        height: 1;
        background: #282828;
    }
    #dp-buttons Button {
        margin: 0 1;
        min-width: 12;
        height: 1;
        min-height: 0;
        border: none;
    }
    #btn-dp-cancel {
        background: #665c54;
        color: #ebdbb2;
    }
    #btn-dp-delete {
        background: #d15d5d;
        color: #1d2021;
        text-style: bold;
    }
    #btn-dp-delete:hover {
        background: #ad3e3e;
        color: #ebdbb2;
    }
    #btn-dp-delete:disabled {
        background: #504945;
        color: #7c6f64;
    }
    """

    def __init__(self, profiles_dir: str):
        super().__init__()
        self.profiles_dir = profiles_dir
        self.selected_profile = ""

    def compose(self) -> ComposeResult:
        with Container(id="delete-profile-container"):
            yield Static("Delete Profile", id="dp-title")
            with VerticalScroll(id="dp-list"):
                yield from self._build_entries()
            with Horizontal(id="dp-buttons"):
                yield Button("Cancel", id="btn-dp-cancel", classes="modal-btn")
                yield Button("Delete", id="btn-dp-delete", variant="error", disabled=True, classes="modal-btn")

    def _build_entries(self):
        try:
            files = sorted(os.listdir(self.profiles_dir))
            custom_profiles_count = 0
            for f in files:
                if f.endswith(".json"):
                    name = f[:-5]
                    if name != "default":
                        custom_profiles_count += 1
                        lbl = Label(f"📄 {name}", classes="dp-entry")
                        lbl.profile_name = name
                        yield lbl
            if custom_profiles_count == 0:
                yield Label("No custom profiles found.", classes="dp-entry")
        except Exception:
            yield Label("⚠ Failed to load profiles", classes="dp-entry")

    def _refresh_list(self) -> None:
        self.selected_profile = ""
        try:
            self.query_one("#btn-dp-delete", Button).disabled = True
        except Exception:
            pass
        try:
            dp_list = self.query_one("#dp-list", VerticalScroll)
            dp_list.remove_children()
            dp_list.mount(*list(self._build_entries()))
        except Exception:
            pass

    def on_click(self, event) -> None:
        widget = event.control if hasattr(event, 'control') else None
        if widget is None or not isinstance(widget, Label):
            return
        if hasattr(widget, "profile_name"):
            profile_name = widget.profile_name
            self.selected_profile = profile_name
            
            try:
                for lbl in self.query(".dp-entry"):
                    lbl.remove_class("selected")
            except Exception:
                pass
            
            widget.add_class("selected")
            
            try:
                self.query_one("#btn-dp-delete", Button).disabled = False
            except Exception:
                pass

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-dp-cancel":
            self.dismiss()
        elif event.button.id == "btn-dp-delete":
            if self.selected_profile:
                self.app.push_screen(
                    ConfirmDeleteModal(self.selected_profile),
                    callback=self._on_delete_confirmed
                )

    def _on_delete_confirmed(self, confirmed: bool) -> None:
        if confirmed and self.selected_profile:
            profile_name = self.selected_profile
            try:
                profile_path = os.path.join(self.profiles_dir, f"{profile_name}.json")
                if os.path.exists(profile_path):
                    os.remove(profile_path)
                
                # If deleted active profile, revert back to default safely
                if self.app.active_profile == profile_name:
                    self.app.active_profile = "default"
                    default_path = os.path.join(self.profiles_dir, "default.json")
                    if os.path.exists(default_path):
                        with open(default_path, "r") as f:
                            default_data = json.load(f)
                        self.app.load_profile_data(default_data)
                    self.app.save_config()
                    self.app.update_profile_loaded_label()

                self.app.show_notification(f"🗑 Profile '{profile_name}' deleted successfully 🗑", severity="success")
                self._refresh_list()
            except Exception as e:
                self.app.show_notification(f"⚠ Error deleting profile: {e} ⚠", severity="error")

    def action_cancel(self) -> None:
        self.dismiss()


class AlertModal(BaseModal[None]):
    """A beautiful centered modal dialog for alerts and notifications."""

    BINDINGS = [
        ("escape", "dismiss_modal", "Dismiss"),
        ("enter", "dismiss_modal", "Dismiss"),
        ("space", "dismiss_modal", "Dismiss"),
    ]

    DEFAULT_CSS = """
    #alert-container {
        width: 50;
        height: 11;
        background: #282828;
        border: thick #d97757;
        padding: 1 2;
        align: center middle;
    }
    #alert-title {
        color: #d97757;
        text-style: bold;
        text-align: center;
        width: 100%;
        margin-bottom: 1;
        background: #282828;
    }
    #alert-message {
        color: #ebdbb2;
        text-align: center;
        width: 100%;
        margin-bottom: 1;
        background: #282828;
    }
    #alert-buttons {
        align: center middle;
        height: 1;
        background: #282828;
    }
    #btn-alert-ok {
        background: #788c5d;
        color: #282828;
        text-style: bold;
        min-width: 12;
        align: center middle;
        height: 1;
        min-height: 0;
        border: none;
        margin: 0;
    }
    #btn-alert-ok:hover {
        background: #586b3e;
        color: #282828;
    }
    """

    def __init__(self, title: str, message: str):
        super().__init__()
        self.title_text = title
        self.message_text = message

    def compose(self) -> ComposeResult:
        with Container(id="alert-container"):
            yield Static(f"✨ {self.title_text} ✨", id="alert-title")
            yield Static(self.message_text, id="alert-message")
            with Horizontal(id="alert-buttons"):
                btn_ok = Button("OK", id="btn-alert-ok")
                btn_ok.styles.height = 1
                btn_ok.styles.min_height = 0
                btn_ok.styles.border = ("none", "transparent")
                btn_ok.styles.margin = (0, 1, 0, 1)
                yield btn_ok

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-alert-ok":
            self.dismiss()

    def action_dismiss_modal(self) -> None:
        self.dismiss()


class AutoDismissModal(BaseModal[None]):
    """A beautiful centered modal that automatically dismisses after 3 seconds."""

    BINDINGS = [
        ("escape", "dismiss_modal", "Dismiss"),
        ("enter", "dismiss_modal", "Dismiss"),
        ("space", "dismiss_modal", "Dismiss"),
    ]

    def __init__(self, title: str, message: str):
        super().__init__()
        self.title_text = title
        self.message_text = message
        self._timer = None

    def compose(self) -> ComposeResult:
        with Container(id="autodismiss-container"):
            yield Static(f"✨ {self.title_text} ✨", id="autodismiss-title")
            yield Static(self.message_text, id="autodismiss-message")

    def on_mount(self) -> None:
        self._timer = self.set_timer(1.8, self.auto_dismiss)

    def on_unmount(self) -> None:
        if self._timer:
            self._timer.stop()

    def auto_dismiss(self) -> None:
        self.dismiss()

    def on_click(self) -> None:
        self.dismiss()

    def action_dismiss_modal(self) -> None:
        self.dismiss()


# NotificationModal replaced with elegant inline notification banner


class LlamaConfigApp(App):
    TITLE = "Llama Launcher"

    # Disable command palette by removing command_palette binding
    BINDINGS = [
        ("ctrl+c", "quit", "Quit")
    ]
    ENABLE_COMMAND_PALETTE = False

    CSS = """
    $surface: #1d2021;
    $boost: #1d2021;
    $panel: #282828;
    $background: #1d2021;

    Screen {
        background: #1d2021;
        padding: 0;
        margin: 0;
        border: none;
        overflow: hidden;
    }

    BaseModal {
        align: center middle;
        background: rgba(0, 0, 0, 0.55);
    }

    #header-container {
        dock: top;
        width: 100%;
        height: auto;
        background: #1d2021;
        layout: vertical;
        align: center middle;
        margin-bottom: 1;
    }
    #ascii-header {
        width: 100%;
        background: transparent;
        color: #d97757;
        text-align: center;
        padding: 0;
    }
    #nav-dropdown-wrapper {
        width: 100%;
        height: 1;
        align: center middle;
        background: transparent;
        margin-bottom: 0;
    }
    #nav-dropdown {
        width: 45;
        height: 1;
        border: none;
    }
    #nav-dropdown SelectCurrent {
        width: 100%;
        background: #3c3836;
        color: #ebdbb2;
        border: none;
        height: 1;
        text-style: bold;
    }
    #nav-dropdown SelectCurrent Static#label {
        width: 1fr;
    }
    #nav-dropdown > SelectOverlay {
        width: 1fr;
    }
    #notification-banner {
        dock: top;
        width: 100%;
        height: 1;
        background: #282828;
        color: #6a9bcc;
        text-align: center;
        text-style: bold;
        display: none;
    }
    #notification-banner.success {
        color: #788c5d;
    }
    #notification-banner.error {
        color: #d15d5d;
    }
    #main-scroll {
        padding: 0 2 0 2;
        background: #1d2021;
    }
    #dashboard-container {
        margin-right: 2;
        margin-bottom: 1;
        background: #282828;
        padding: 1 2;
        height: auto;
        layout: vertical;
    }
    #dashboard-title-label {
        color: #d97757;
        text-style: bold;
        margin-bottom: 1;
    }
    #vram-meter-container {
        height: 1;
        background: transparent;
        margin-bottom: 0;
    }
    #vram-title {
        color: #ebdbb2;
        text-style: bold;
    }
    #vram-text {
        color: #ebdbb2;
        text-style: bold;
    }
    #vram-bar {
        background: transparent;
        color: #788c5d;
        text-style: bold;
        margin-bottom: 1;
        width: 100%;
    }
    #preview-header {
        height: 1;
        background: transparent;
        align: left middle;
        margin-bottom: 0;
    }
    #preview-title {
        color: #d97757;
        text-style: bold;
        width: 1fr;
    }
    #preview-hint {
        color: #7c6f64;
        text-style: dim;
        margin: 0;
        padding: 0;
    }

    #command-preview {
        background: #1d2021;
        color: #ebdbb2;
        padding: 0 1;
        height: auto;
        border: none;
        margin-top: 0;
        width: 100%;
        content-align: left top;
        overflow-x: scroll;
        overflow-y: hidden;
    }
    Input:focus {
        background: #6a9bcc;
        color: #1d2021;
    }
    Select:focus SelectCurrent, SelectCurrent:focus {
        background: #6a9bcc;
        color: #1d2021;
    }
    Label {
        color: #ebdbb2;
        text-style: bold;
    }
    .custom-checkbox {
        background: #282828;
        color: #788c5d;
        text-style: bold;
        width: 3;
        margin-right: 1;
        content-align: center middle;
    }
    .custom-checkbox.dimmed {
        color: #7c6f64;
    }
    .field-label.dimmed {
        color: #7c6f64;
    }
    .info-icon.dimmed {
        color: #504945;
    }
    Input.dimmed {
        background: #1d2021;
        color: #7c6f64;
        border: none;
    }
    Select.dimmed SelectCurrent {
        background: #1d2021;
        color: #7c6f64;
        border: none;
    }
    .browse-btn.dimmed {
        background: #1d2021;
        color: #7c6f64;
        border: none;
    }

    Button {
        margin: 2 2;
        background: #3c3836;
        color: #ebdbb2;
        border: none;
        text-style: bold;
    }
    Button:hover {
        background: #504945;
        color: #fbf1c7;
    }
    .modal-btn {
        margin: 0 1;
        min-width: 12;
        height: 1;
        min-height: 0;
        border: none;
    }
    #buttons {
        dock: bottom;
        layout: vertical;
        height: 9;
        background: #1d2021;
        border-top: solid #3c3836;
        align: center middle;
        padding-top: 0;
    }
    #profile-loaded-label {
        color: #a89984;
        text-align: center;
        width: 100%;
        height: 1;
        margin-bottom: 1;
    }
    #profile-buttons {
        align: center middle;
        height: 1;
        margin-bottom: 1;
    }
    .profile-btn {
        margin: 0 2;
        height: 1;
        min-height: 0;
    }
    #start-button-container {
        align: center middle;
        height: 4;
    }
    #btn-load-profile {
        background: #d97757;
        color: #1d2021;
        text-style: bold;
    }
    #btn-load-profile:hover {
        background: #b85b3b;
        color: #fbf1c7;
    }
    #btn-save-profile {
        background: #d97757;
        color: #1d2021;
        text-style: bold;
    }
    #btn-save-profile:hover {
        background: #b85b3b;
        color: #fbf1c7;
    }
    #btn-delete-profile {
        background: #d97757;
        color: #1d2021;
        text-style: bold;
    }
    #btn-delete-profile:hover {
        background: #b85b3b;
        color: #fbf1c7;
    }
    #btn-refresh-vram {
        background: #d97757;
        color: #1d2021;
        text-style: bold;
    }
    #btn-refresh-vram:hover {
        background: #b85b3b;
        color: #fbf1c7;
    }
    #btn-start {
        background: #6a9bcc;
        color: #1d2021;
        text-style: bold;
        height: 3;
        min-height: 3;
        width: 24;
        min-width: 24;
        border: none;
        content-align: center middle;
        margin: 0 0 1 0;
    }
    #btn-start:hover {
        background: #4a7bab;
        color: #fbf1c7;
    }
    .grid-container {
        layout: horizontal;
        height: auto;
        background: #282828;
    }
    .grid-column {
        width: 1fr;
        height: auto;
        background: #282828;
        padding-right: 4;
    }
    ParameterField {
        layout: horizontal;
        height: 1;
        align: left middle;
        background: #282828;
        margin-bottom: 1;
    }
    .field-label {
        width: 25;
        background: #282828;
    }
    Input {
        width: 1fr;
        background: #3c3836;
        border: none;
        color: #ebdbb2;
        padding: 0 1;
        height: 1;
        margin-bottom: 0;
    }
    Select {
        width: auto;
        height: 1;
        border: none;
        background: transparent;
        padding: 0;
        margin: 0;
    }
    SelectCurrent {
        width: auto;
        min-width: 8;
        background: #3c3836;
        color: #ebdbb2;
        border: none;
        height: 1;
        padding: 0 1;
    }
    SelectCurrent .arrow {
        padding: 0 0 0 1;
    }
    SelectCurrent Static#label {
        width: auto;
    }
    SelectOverlay {
        width: auto;
        padding: 0;
    }
    .spacer-checkbox {
        width: 3;
        margin-right: 1;
        background: #282828;
    }
    .section-card {
        margin-right: 2;
        margin-bottom: 2;
        background: #282828;
        border: none;
        padding: 1 2;
        height: auto;
        layout: vertical;
    }
    .section-card-title {
        color: #d97757;
        text-style: bold;
        background: transparent;
        margin-bottom: 1;
        width: 100%;
        text-align: left;
    }

    .dimmed {
    }
    .info-icon {
        width: 3;
        color: #a89984;
        background: transparent;
        text-style: bold;
        content-align: center middle;
    }
    .info-icon:hover {
        color: #6a9bcc;
    }
    .browse-btn {
        min-width: 4;
        width: 4;
        height: 1;
        margin: 0 0 0 1;
        padding: 0;
        background: #3c3836;
        color: #6a9bcc;
        border: none;
        text-style: bold;
    }
    .browse-btn:hover {
        background: #6a9bcc;
        color: #1d2021;
    }
    Tooltip {
        background: #3c3836;
        color: #ebdbb2;
        border: solid #6a9bcc;
        padding: 1 2;
        max-width: 50;
    }
    Toast {
        background: #282828;
        color: #fbf1c7;
        border: solid #6a9bcc;
        padding: 1 2;
        height: auto;
        min-height: 3;
    }
    Toast:hover {
        background: #3c3836;
    }
    Toast .toast--title {
        color: #6a9bcc;
        text-style: bold;
    }
    ToastRack {
        align: right top;
        dock: right;
        layer: _toast;
        width: auto;
        height: auto;
        overflow: hidden;
        margin: 1 2;
    }
    AutoDismissModal {
        align: center middle;
        background: transparent !important;
    }
    #autodismiss-container {
        width: 60;
        height: 8;
        background: #282828;
        border: thick #d97757;
        padding: 1 2;
        align: center middle;
    }
    #autodismiss-title {
        color: #d97757;
        text-style: bold;
        text-align: center;
        width: 100%;
        margin-bottom: 1;
        background: #282828;
    }
    #autodismiss-message {
        color: #ebdbb2;
        text-align: center;
        width: 100%;
        background: #282828;
    }
    """

    def __init__(self):
        super().__init__()
        self.config = DEFAULT_CONFIG.copy()
        self.should_start = False
        self.enabled_fields = {}
        self.fields_by_key = {}
        self._debounce_timer = None
        
        # Ensure profiles directory exists
        self.profiles_dir = os.path.join(CONFIG_DIR, "profiles")
        os.makedirs(self.profiles_dir, exist_ok=True)
        self.active_profile = "default"
        profile_data = {}
        
        # Load initial config
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, "r") as f:
                    loaded = json.load(f)
                    self.config.update(loaded)
                    self.active_profile = loaded.get("ACTIVE_PROFILE", "default")
            except Exception:
                pass
                
        # Load active profile JSON if it exists
        active_profile_path = os.path.join(self.profiles_dir, f"{self.active_profile}.json")
        if os.path.exists(active_profile_path):
            try:
                with open(active_profile_path, "r") as f:
                    profile_data = json.load(f)
                    self.config.update(profile_data)
            except Exception:
                pass
        
        # Initialize default.json profile if it doesn't exist
        default_profile_path = os.path.join(self.profiles_dir, "default.json")
        if not os.path.exists(default_profile_path):
            try:
                with open(default_profile_path, "w") as f:
                    json.dump(self.config, f, indent=4)
            except Exception:
                pass
                
        # Determine enabled fields
        disabled = self.config.get("DISABLED_FIELDS", [])
        if not disabled and not os.path.exists(CONFIG_FILE):
            disabled = [
                "DRY_MULTIPLIER", "DRY_BASE", "DRY_ALLOWED_LENGTH", "DRY_PENALTY_LAST_N",
                "XTC_PROBABILITY", "XTC_THRESHOLD"
            ]
        for k in DEFAULT_CONFIG:
            is_mandatory = k in ("BIN", "MODEL", "HOST", "PORT", "EXTRA_FLAGS")
            if is_mandatory:
                self.enabled_fields[k] = True
            else:
                val = str(self.config.get(k, "")).strip()
                if val == "":
                    self.enabled_fields[k] = False
                else:
                    self.enabled_fields[k] = k not in disabled

    def _load_gguf_metadata_worker(self, model_path: str) -> None:
        """Worker thread entry point to parse GGUF file without locking main thread."""
        info = {"size_gib": 16.5, "metadata": {}, "parsed": True}
        if os.path.isfile(model_path):
            try:
                info["size_gib"] = os.path.getsize(model_path) / (1024 ** 3)
                info["metadata"] = parse_gguf_metadata(model_path)
            except Exception:
                pass
        
        # Pre-parse parameters directly in the background worker
        info["params"] = pre_parse_gguf_params(info["metadata"], model_path)
        
        with GGUF_LOCK:
            GGUF_CACHE[model_path] = info
            if model_path in PENDING_GGUF_LOADS:
                PENDING_GGUF_LOADS.remove(model_path)
        
        # Thread-safe UI update trigger
        self.call_from_thread(self.update_dashboard)

    def debounce_update_dashboard(self) -> None:
        """Buffer updates during fast keyboard typing to remove input stutters."""
        if self._debounce_timer is not None:
            try:
                self._debounce_timer.stop()
            except Exception:
                pass
        self._debounce_timer = self.set_timer(0.35, self.update_dashboard)

    def get_css_variables(self) -> dict[str, str]:
        variables = super().get_css_variables()
        bg = "#282828"
        for key in ("surface", "panel", "boost", "background",
                    "surface-active", "block-hover-background",
                    "panel-darken-1", "panel-darken-2", "panel-darken-3",
                    "panel-lighten-1", "panel-lighten-2", "panel-lighten-3",
                    "surface-darken-1", "surface-darken-2", "surface-darken-3",
                    "surface-lighten-1", "surface-lighten-2", "surface-lighten-3",
                    "boost-darken-1", "boost-darken-2", "boost-darken-3",
                    "boost-lighten-1", "boost-lighten-2", "boost-lighten-3"):
            variables[key] = bg
        return variables

    def show_notification(self, message: str, severity: str = "success") -> None:
        title = "Notification"
        if severity == "error":
            title = "Error"
        elif severity == "success":
            title = "Success"
        self.push_screen(AutoDismissModal(title, message))

    def hide_notification(self) -> None:
        pass

    def compose(self) -> ComposeResult:
        nav_options = [
            ("Infrastructure", "card-infra"),
            ("Hardware & Performance", "card-hardware"),
            ("Context & Cache", "card-context"),
            ("HTTP Server", "card-http"),
            ("Sampling", "card-sampling"),
            ("Advanced Samplers", "card-advanced"),
            ("Speculative Decoding (MTP)", "card-speculative"),
            ("Reasoning & Agentic", "card-reasoning"),
            ("Monitoring & Logging", "card-features"),
            ("Multimodal & Templates", "card-multimodal"),
            ("Custom Parameters", "card-custom"),
        ]
        with Vertical(id="header-container"):
            yield Static(ASCII_HEADER, id="ascii-header")
            with Horizontal(id="nav-dropdown-wrapper"):
                yield Select(nav_options, prompt="🔍 Jump to Section...", id="nav-dropdown", allow_blank=True)
        
        sections = [
            ("Infrastructure", [
                ("BIN", "Server Binary:"),
                ("MODEL", "Model Path:"),
                ("HOST", "Host IP:"),
                ("PORT", "Port:"),
                ("API_KEY", "API Key:"),
                ("ALIAS", "Model Alias:"),
                ("WEBUI", "Web UI:"),
            ]),
            ("Hardware & Performance", [
                ("NGL", "GPU Layers:"),
                ("THREADS", "CPU Threads:"),
                ("THREADS_BATCH", "Batch Threads:"),
                ("BATCH_SIZE", "Batch Size:"),
                ("UBATCH_SIZE", "Micro Batch:"),
                ("FLASH_ATTN", "Flash Attention:"),
                ("MLOCK", "MLock:"),
                ("NO_MMAP", "Disable MMAP:"),
                ("SPLIT_MODE", "Split Mode:"),
                ("MAIN_GPU", "Main GPU:"),
                ("TENSOR_SPLIT", "Tensor Split:"),
                ("NUMA", "NUMA Optimization:"),
                ("GGML_CUDA_DISABLE_GRAPHS", "Disable CUDA Graphs:"),
                ("GGML_CUDA_ENABLE_UNIFIED_MEMORY", "Unified Memory:"),
            ]),
            ("Context & Cache", [
                ("CTX", "Context Size:"),
                ("NP", "Parallel Slots:"),
                ("CACHE_K", "KV Cache K:"),
                ("CACHE_V", "KV Cache V:"),
                ("CTX_CHECKPOINTS", "Ctx Checkpoints:"),
                ("SWA_FULL", "SWA Full Cache:"),
                ("CACHE_RAM", "Cache RAM MB:"),
                ("CONT_BATCHING", "Cont Batching:"),
                ("CACHE_PROMPT", "Prompt Caching:"),
                ("CACHE_REUSE", "Cache Reuse Min:"),
            ]),
            ("HTTP Server", [
                ("TIMEOUT", "Server Timeout:"),
                ("THREADS_HTTP", "HTTP Threads:"),
            ]),
            ("Sampling", [
                ("TEMP", "Temperature:"),
                ("TOP_P", "Top-P:"),
                ("TOP_K", "Top-K:"),
                ("MIN_P", "Min-P:"),
                ("PRESENCE_PENALTY", "Presence Pen:"),
                ("REPEAT_PENALTY", "Repeat Pen:"),
                ("PREDICT", "Predict Limit:"),
                ("SEED", "Seed:"),
            ]),
            ("Advanced Samplers", [
                ("DRY_MULTIPLIER", "DRY Multiplier:"),
                ("DRY_BASE", "DRY Base Value:"),
                ("DRY_ALLOWED_LENGTH", "DRY Allowed Len:"),
                ("DRY_PENALTY_LAST_N", "DRY Last N:"),
                ("XTC_PROBABILITY", "XTC Probability:"),
                ("XTC_THRESHOLD", "XTC Threshold:"),
            ]),
            ("Speculative Decoding (MTP)", [
                ("MODEL_DRAFT", "Draft Model:"),
                ("SPEC_TYPE", "Spec Type:"),
                ("SPEC_MAX", "Spec Max Draft:"),
                ("SPEC_MIN", "Spec Min Prob:"),
                ("SPEC_DRAFT_N_MIN", "Spec Draft N Min:"),
                ("DRAFT_MAX", "Draft Max Steps:"),
                ("DRAFT_P_MIN", "Draft Min Prob:"),
            ]),
            ("Reasoning & Agentic", [
                ("REASONING", "Thinking Mode:"),
                ("REASONING_FORMAT", "Thinking Format:"),
                ("REASONING_BUDGET", "Thinking Budget:"),
                ("TOOLS", "Agentic Tools:"),
            ]),
            ("Monitoring & Logging", [
                ("METRICS", "Metrics:"),
            ]),
            ("Multimodal & Templates", [
                ("JINJA", "Jinja Templates:"),
                ("MMPROJ", "Vision Projector:"),
                ("MMPROJ_OFFLOAD", "GPU MM Offload:"),
                ("IMAGE_MIN_TOKENS", "Image Min Tokens:"),
                ("IMAGE_MAX_TOKENS", "Image Max Tokens:"),
                ("CHAT_TEMPLATE", "Chat Template:"),
                ("CHAT_TEMPLATE_FILE", "Template File:"),
                ("TEMPLATE_KWARGS", "Template Kwargs:"),
                ("SKIP_CHAT_PARSING", "Skip Chat Parse:"),
            ]),
            ("Custom Parameters", [
                ("EXTRA_FLAGS", "Other Parameters:"),
            ]),
        ]

        with VerticalScroll(id="main-scroll"):
            for title, fields in sections:
                class_map = {
                    "Infrastructure": "card-infra",
                    "Hardware & Performance": "card-hardware",
                    "Context & Cache": "card-context",
                    "HTTP Server": "card-http",
                    "Sampling": "card-sampling",
                    "Advanced Samplers": "card-advanced",
                    "Speculative Decoding (MTP)": "card-speculative",
                    "Reasoning & Agentic": "card-reasoning",
                    "Monitoring & Logging": "card-features",
                    "Multimodal & Templates": "card-multimodal",
                    "Custom Parameters": "card-custom"
                }
                card_class = class_map.get(title, "card-default")
                with Vertical(classes=f"section-card {card_class}"):
                    yield Label(f" {title.upper()} ", classes="section-card-title")
                    for key, label in fields:
                        is_mandatory = key in ("BIN", "MODEL", "HOST", "PORT", "EXTRA_FLAGS")
                        is_enabled = self.enabled_fields.get(key, True)
                        yield ParameterField(
                            key=key, 
                            label=label, 
                            value=str(self.config.get(key, "")),
                            is_mandatory=is_mandatory,
                            is_enabled=is_enabled
                        )

            with Container(id="dashboard-container"):
                yield Label("📊 REAL-TIME DASHBOARD", id="dashboard-title-label")
                with Horizontal(id="vram-meter-container"):
                    yield Label("EST VRAM USAGE: ", id="vram-title")
                    yield Label("0.0 GB / 24 GB (0%)", id="vram-text")
                yield Label("[░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]", id="vram-bar")
                
                with Horizontal(id="preview-header"):
                    yield Label("LIVE BASH COMMAND PREVIEW:", id="preview-title")
                    yield Label("Double-click inside to select launch command", id="preview-hint")
                yield Static("", id="command-preview")

        with Vertical(id="buttons"):
            yield Label("", id="profile-loaded-label")
            with Horizontal(id="profile-buttons"):
                yield Button("Load Profile", id="btn-load-profile", variant="default", classes="profile-btn")
                yield Button("Save profile as...", id="btn-save-profile", variant="primary", classes="profile-btn")
                yield Button("Delete Profile", id="btn-delete-profile", variant="error", classes="profile-btn")
                yield Button("Refresh VRAM", id="btn-refresh-vram", variant="warning", classes="profile-btn")
            with Horizontal(id="start-button-container"):
                yield Button("Start Server", id="btn-start", variant="success")

    @on(Select.Changed)
    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "nav-dropdown" and event.value != Select.BLANK:
            target_class = event.value
            try:
                scroll_container = self.query_one("#main-scroll", VerticalScroll)
                target_card = scroll_container.query_one(f".{target_class}")
                scroll_container.scroll_to_widget(target_card)
                # Reset dropdown to blank so it shows the prompt and can be reused
                event.select.value = Select.BLANK
            except Exception:
                pass

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-load-profile":
            self.push_screen(
                LoadProfileModal(profiles_dir=self.profiles_dir),
                callback=self._on_profile_loaded
            )
        elif event.button.id == "btn-save-profile":
            self.push_screen(
                SaveProfileModal(profiles_dir=self.profiles_dir, current_name=self.active_profile),
                callback=self._on_profile_saved_as
            )
        elif event.button.id == "btn-delete-profile":
            self.push_screen(
                DeleteProfileModal(profiles_dir=self.profiles_dir),
                callback=self._on_profile_deleted
            )
        elif event.button.id == "btn-start":
            self.save_config()
            self.should_start = True
            self.exit()
        elif event.button.id == "btn-refresh-vram":
            defrag_script = os.path.join(CONFIG_DIR, "defrag_vram.py")
            if os.path.exists(defrag_script):
                self.push_screen(AutoDismissModal("VRAM Defrag", "VRAM successfully defragmented!"))
                # Execute defrag script in a background thread to prevent UI blocking
                self.run_worker(self.sweep_vram_background, thread=True)
        elif event.button.id == "btn-browse-model":
            field = self.fields_by_key.get("MODEL")
            if field is not None:
                self.push_screen(
                    FileBrowserModal(start_path=field.value),
                    callback=self._on_model_selected,
                )
        elif event.button.id == "btn-browse-model-draft":
            field = self.fields_by_key.get("MODEL_DRAFT")
            if field is not None:
                self.push_screen(
                    FileBrowserModal(start_path=field.value),
                    callback=self._on_model_draft_selected,
                )
        elif event.button.id == "btn-browse-bin":
            field = self.fields_by_key.get("BIN")
            if field is not None:
                self.push_screen(
                    FileBrowserModal(start_path=field.value, file_extensions=None),
                    callback=self._on_bin_selected,
                )
        elif event.button.id == "btn-browse-mmproj":
            field = self.fields_by_key.get("MMPROJ")
            if field is not None:
                self.push_screen(
                    FileBrowserModal(start_path=field.value),
                    callback=self._on_mmproj_selected,
                )
        elif event.button.id == "btn-browse-template":
            field = self.fields_by_key.get("CHAT_TEMPLATE_FILE")
            if field is not None:
                self.push_screen(
                    FileBrowserModal(start_path=field.value, file_extensions=(".jinja", ".txt", ".tmpl")),
                    callback=self._on_template_selected,
                )



    def sweep_vram_background(self) -> None:
        defrag_script = os.path.join(CONFIG_DIR, "defrag_vram.py")
        if os.path.exists(defrag_script):
            try:
                subprocess.run(
                    ["python3", defrag_script],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False
                )
            except Exception:
                pass

    def _on_model_selected(self, path: str) -> None:
        if path:
            field = self.fields_by_key.get("MODEL")
            if field is not None:
                field.value = path
            self.update_dashboard()

    def _on_model_draft_selected(self, path: str) -> None:
        if path:
            field = self.fields_by_key.get("MODEL_DRAFT")
            if field is not None:
                field.value = path
            self.update_dashboard()

    def _on_bin_selected(self, path: str) -> None:
        if path:
            field = self.fields_by_key.get("BIN")
            if field is not None:
                field.value = path
            self.update_dashboard()

    def _on_mmproj_selected(self, path: str) -> None:
        if path:
            field = self.fields_by_key.get("MMPROJ")
            if field is not None:
                field.value = path
            self.update_dashboard()

    def _on_template_selected(self, path: str) -> None:
        if path:
            field = self.fields_by_key.get("CHAT_TEMPLATE_FILE")
            if field is not None:
                field.value = path
            self.update_dashboard()

    def _on_profile_loaded(self, profile_name: str) -> None:
        if profile_name:
            profile_path = os.path.join(self.profiles_dir, f"{profile_name}.json")
            if os.path.exists(profile_path):
                try:
                    with open(profile_path, "r") as f:
                        data = json.load(f)
                    self.load_profile_data(data)
                    self.active_profile = profile_name
                    self.save_config()  # Dual-saves to update ACTIVE_PROFILE in config.json
                    self.update_profile_loaded_label()
                    self.show_notification(f"✨ Profile '{profile_name}' loaded successfully! ✨", severity="success")
                except Exception as e:
                    self.show_notification(f"⚠ Error loading profile: {e} ⚠", severity="error")

    def _on_profile_saved_as(self, profile_name: str) -> None:
        if profile_name:
            self.active_profile = profile_name
            self.save_config()  # Dual-saves to both config.json and the new profile
            self.update_profile_loaded_label()
            self.show_notification(f"✨ Profile '{profile_name}' saved! ✨", severity="success")

    def _on_profile_deleted(self, result) -> None:
        pass

    def update_profile_loaded_label(self) -> None:
        try:
            lbl = self.query_one("#profile-loaded-label", Label)
            lbl.update(f"Profile Loaded: [bold #d97757]{self.active_profile}[/]")
        except Exception:
            pass

    def load_profile_data(self, data: dict) -> None:
        # Normalize configuration keys to uppercase
        data = {k.upper(): v for k, v in data.items()}
        disabled = [item.upper() for item in data.get("DISABLED_FIELDS", []) if isinstance(item, str)]
        
        new_config = DEFAULT_CONFIG.copy()
        new_config.update(data)
        
        for k, v in new_config.items():
            if k in SELECT_OPTIONS:
                new_config[k] = normalize_select_value(k, v)
                        
        self.config = new_config
        
        fields = self._parameter_fields

        if fields:
            for field in fields:
                is_mandatory = field.key in ("BIN", "MODEL", "HOST", "PORT", "EXTRA_FLAGS")
                if field.key in data:
                    val = str(data[field.key]).strip()
                    field.value = val
                    if is_mandatory:
                        field.enabled = True
                    else:
                        if val == "":
                            field.enabled = False
                        else:
                            field.enabled = field.key not in disabled
                else:
                    if is_mandatory:
                        field.enabled = True
                    else:
                        field.value = str(DEFAULT_CONFIG.get(field.key, ""))
                        field.enabled = False
                field.update_visuals()
        else:
            for k in DEFAULT_CONFIG:
                is_mandatory = k in ("BIN", "MODEL", "HOST", "PORT", "EXTRA_FLAGS")
                if is_mandatory:
                    self.enabled_fields[k] = True
                else:
                    if k in data:
                        val = str(data[k]).strip()
                        if val == "":
                            self.enabled_fields[k] = False
                        else:
                            self.enabled_fields[k] = k not in disabled
                    else:
                        self.enabled_fields[k] = False

        self.update_dashboard()

    def save_config(self):
        disabled = []
        fields = self._parameter_fields

        if fields:
            for field in fields:
                self.config[field.key] = field.value
                self.enabled_fields[field.key] = field.enabled
                if not field.enabled:
                    disabled.append(field.key)
            self.config["DISABLED_FIELDS"] = disabled
        else:
            self.config["DISABLED_FIELDS"] = [k for k, enabled in self.enabled_fields.items() if not enabled]

        self.config["ACTIVE_PROFILE"] = self.active_profile
        
        # Save active session config
        with open(CONFIG_FILE, "w") as f:
            json.dump(self.config, f, indent=4)
            
        # Save to current active profile JSON
        profile_path = os.path.join(self.profiles_dir, f"{self.active_profile}.json")
        try:
            with open(profile_path, "w") as f:
                json.dump(self.config, f, indent=4)
        except Exception as e:
            self.show_notification(f"⚠ Failed to save profile file: {e} ⚠", severity="error")

    @property
    def _parameter_fields(self) -> list[ParameterField]:
        if not hasattr(self, "_cached_parameter_fields") or not self._cached_parameter_fields:
            fields = []
            try:
                if hasattr(self, "screen_stack") and self.screen_stack:
                    fields = list(self.screen_stack[0].query(ParameterField))
            except Exception as e:
                logging.debug(f"[DEBUG _parameter_fields screen_stack error]: {e}")
            if not fields:
                try:
                    fields = list(self.query(ParameterField))
                except Exception as e:
                    logging.debug(f"[DEBUG _parameter_fields query error]: {e}")
                    return []
            self._cached_parameter_fields = fields
        return self._cached_parameter_fields

    @_parameter_fields.setter
    def _parameter_fields(self, val) -> None:
        if isinstance(val, list):
            self._cached_parameter_fields = val

    @property
    def fields_by_key(self) -> dict[str, ParameterField]:
        if not hasattr(self, "_cached_fields_by_key") or not self._cached_fields_by_key:
            self._cached_fields_by_key = {field.key: field for field in self._parameter_fields}
        return self._cached_fields_by_key

    @fields_by_key.setter
    def fields_by_key(self, val) -> None:
        if isinstance(val, dict):
            self._cached_fields_by_key = val

    def on_mount(self) -> None:
        self.update_profile_loaded_label()
        try:
            self.vram_text_widget = self.query_one("#vram-text", Label)
            self.vram_bar_widget = self.query_one("#vram-bar", Label)
            self.cmd_preview_widget = self.query_one("#command-preview", Static)
        except Exception:
            pass
        self.update_dashboard()

    def get_current_active_parameters(self) -> dict[str, str]:
        params = {}
        fields = self._parameter_fields
        
        if fields:
            for field in fields:
                if field.enabled:
                    params[field.key] = field.value
        else:
            for k, v in self.config.items():
                if self.enabled_fields.get(k, True):
                    params[k] = str(v)
        return params

    def calculate_vram_estimate(self, params: dict[str, str]) -> tuple[float, int]:
        # Estimate model weights memory: dynamically read from cached model info
        model_path = params.get("MODEL", "")
        model_info = get_model_info(model_path, app=self)
        base_vram = model_info["size_gib"]
        
        # Speculative draft model weights
        draft_vram = 0.0
        draft_model_path = params.get("MODEL_DRAFT", "")
        if draft_model_path:
            draft_info = get_model_info(draft_model_path, app=self)
            draft_vram = draft_info["size_gib"]
        
        # NGL offload layers
        ngl = 99
        if "NGL" in params:
            try:
                ngl = int(params["NGL"])
            except ValueError:
                ngl = 99
                
        model_offload_ratio = min(ngl / 99.0, 1.0)
        weights_on_gpu = (base_vram + draft_vram) * model_offload_ratio
        
        # Context size
        ctx = 8192
        if "CTX" in params:
            try:
                ctx = int(params["CTX"])
            except ValueError:
                ctx = 8192

        # Parallel slots
        np = 1
        if "NP" in params:
            try:
                np = int(params["NP"])
            except ValueError:
                np = 1
                
        # Flash Attention
        use_flash_attn = True
        if "FLASH_ATTN" in params:
            use_flash_attn = params["FLASH_ATTN"] != "off"
            
        # KV Cache key and value types (bytes per element)
        def get_element_bytes(cache_type: str) -> float:
            cache_type = cache_type.strip().lower()
            if cache_type == "f32":
                return 4.0
            elif cache_type == "f16":
                return 2.0
            elif cache_type == "q8_0":
                return 1.0625  # Incorporates Q8 block metadata scaling overhead (34 bytes for 32 elements)
            elif cache_type in ("q4_0", "q4_1", "q4_k_m", "q4_k_s", "q4_2"):
                return 0.5625  # 18 bytes for 32 elements
            elif cache_type in ("q5_0", "q5_1", "q5_k_m", "q5_k_s"):
                return 0.6875  # 22 bytes for 32 elements
            return 2.0  # default to f16

        cache_k = params.get("CACHE_K", "f16")
        cache_v = params.get("CACHE_V", "f16")
        
        bytes_k = get_element_bytes(cache_k)
        bytes_v = get_element_bytes(cache_v)

        # Pre-parsed parameters cache query in O(1) time complexity
        arch_params = model_info.get("params", {
            "block_count": 32,
            "head_count_kv": 8,
            "embedding_length": 4096,
            "head_count": 32,
            "head_dim": 128
        })
        block_count = arch_params["block_count"]
        head_count_kv = arch_params["head_count_kv"]
        head_dim = arch_params["head_dim"]
        
        # Calculate KV Cache size in GiB
        kv_cache_total_bytes = np * ctx * block_count * head_count_kv * head_dim * (bytes_k + bytes_v)
        kv_memory = kv_cache_total_bytes / (1024 ** 3)
        
        # Flash Attention workspace overhead scaling
        if not use_flash_attn:
            kv_memory *= 1.3
            
        # Speculative decoding memory overhead (MTP is integrated and lightweight)
        speculator_overhead = 0.0
        spec_type = params.get("SPEC_TYPE", "none").strip().lower()
        if spec_type != "none" and spec_type != "":
            if "mtp" in spec_type:
                speculator_overhead = 0.3
            else:
                speculator_overhead = 1.5
            
        # Multimodal Projector overhead
        vision_projector_overhead = 0.0
        if "MMPROJ" in params and params["MMPROJ"].strip():
            vision_projector_overhead = 0.8
            
        # CUDA context initialization overhead
        cuda_overhead = 0.0
        if ngl > 0:
            cuda_overhead = 0.8
            
        total_estimate = weights_on_gpu + kv_memory + speculator_overhead + vision_projector_overhead + cuda_overhead
        total_estimate = min(max(total_estimate, 0.0), 96.0)
        
        percentage = round((total_estimate / 24.0) * 100)
        return total_estimate, percentage

    def build_live_command(self, params: dict[str, str]) -> str:
        cmd = []
        
        # Add environment variables if enabled
        if "GGML_CUDA_DISABLE_GRAPHS" in params and params["GGML_CUDA_DISABLE_GRAPHS"].lower() in ("on", "1", "true", "yes"):
            cmd.append("GGML_CUDA_DISABLE_GRAPHS=1")
            
        if "GGML_CUDA_ENABLE_UNIFIED_MEMORY" in params and params["GGML_CUDA_ENABLE_UNIFIED_MEMORY"].lower() in ("on", "1", "true", "yes"):
            cmd.append("GGML_CUDA_ENABLE_UNIFIED_MEMORY=1")
        
        # Get binary path
        bin_path = params.get("BIN", "~/llama.cpp/build/bin/llama-server")
        bin_path = os.path.expanduser(bin_path)
        cmd.append(bin_path)

        # Build arguments list using helper
        cmd.extend(build_server_args(params))

        # Format arguments with backslash wrapping for readability
        formatted_args = []
        for idx, arg in enumerate(cmd):
            if idx == 0:
                formatted_args.append(arg)
            else:
                if arg.startswith("-") and not (len(arg) > 1 and arg[1].isdigit()):
                    formatted_args.append(f"\\\n  {arg}")
                else:
                    if any(char in arg for char in " []{}()*?&\"'"):
                        formatted_args.append(shlex.quote(arg))
                    else:
                        formatted_args.append(arg)
                        
        return " ".join(formatted_args)

    def update_dashboard(self) -> None:
        if not self.is_mounted:
            return
            
        try:
            params = self.get_current_active_parameters()
            logging.debug(f"[{time.time()}] update_dashboard called. CTX={params.get('CTX', '')}, TEMP={params.get('TEMP', '')}")
            
            # Skip calculations and repaints if parameters haven't changed to optimize UI performance
            if hasattr(self, "_last_active_params") and self._last_active_params == params:
                return
            self._last_active_params = params.copy()
            
            # VRAM estimate
            total_estimate, percentage = self.calculate_vram_estimate(params)
            
            # Update VRAM Text using cached widget references to eliminate DOM search overhead
            vram_text = getattr(self, "vram_text_widget", None)
            if vram_text is None:
                try:
                    vram_text = self.query_one("#vram-text", Label)
                    self.vram_text_widget = vram_text
                except Exception:
                    vram_text = None
            if vram_text is not None:
                vram_text.update(f"{total_estimate:.1f} GB / 24 GB ({percentage}%)")
            
            # Color coding threshold based on saturation
            if percentage > 100:
                if vram_text is not None:
                    vram_text.styles.color = "#d15d5d" # Red
                bar_color = "#d15d5d"
            elif percentage > 90:
                if vram_text is not None:
                    vram_text.styles.color = "#d97757" # Orange
                bar_color = "#d97757"
            elif percentage > 75:
                if vram_text is not None:
                    vram_text.styles.color = "#6a9bcc" # Blue
                bar_color = "#6a9bcc"
            else:
                if vram_text is not None:
                    vram_text.styles.color = "#ebdbb2" # Cream
                bar_color = "#788c5d" # Green

                
            # Build Unicode block meter progress bar (40 character width)
            # █ (solid block) and ░ (light shade block)
            filled_chars = min(max(round((percentage / 100.0) * 40), 0), 40)
            empty_chars = 40 - filled_chars
            meter_str = f"[{'█' * filled_chars}{'░' * empty_chars}]"
            
            vram_bar = getattr(self, "vram_bar_widget", None)
            if vram_bar is None:
                try:
                    vram_bar = self.query_one("#vram-bar", Label)
                    self.vram_bar_widget = vram_bar
                except Exception:
                    vram_bar = None
            if vram_bar is not None:
                vram_bar.update(meter_str)
                vram_bar.styles.color = bar_color
            
            # Update Live command preview using cached widget reference
            cmd_preview = getattr(self, "cmd_preview_widget", None)
            if cmd_preview is None:
                try:
                    cmd_preview = self.query_one("#command-preview", Static)
                    self.cmd_preview_widget = cmd_preview
                except Exception:
                    cmd_preview = None
            if cmd_preview is not None:
                live_command = self.build_live_command(params)
                cmd_preview.update(live_command)
            
        except Exception:
            pass


if __name__ == "__main__":
    app = LlamaConfigApp()
    app.run()

    if app.should_start:
        os.system("clear")
        print("Starting Llama Server...")
        cfg = app.config
        
        # CUDA Graphs toggle:
        disable_graphs = cfg.get("GGML_CUDA_DISABLE_GRAPHS", "on")
        disable_graphs_enabled = app.enabled_fields.get("GGML_CUDA_DISABLE_GRAPHS", True)
        if disable_graphs_enabled and disable_graphs.lower() in ("on", "1", "true", "yes"):
            os.environ["GGML_CUDA_DISABLE_GRAPHS"] = "1"
            print("CUDA Graphs: DISABLED (via GGML_CUDA_DISABLE_GRAPHS=1)")
        else:
            os.environ.pop("GGML_CUDA_DISABLE_GRAPHS", None)
            print("CUDA Graphs: ENABLED")
        
        # Unified Memory toggle:
        unified_memory = cfg.get("GGML_CUDA_ENABLE_UNIFIED_MEMORY", "off")
        unified_memory_enabled = app.enabled_fields.get("GGML_CUDA_ENABLE_UNIFIED_MEMORY", True)
        if unified_memory_enabled and unified_memory.lower() in ("on", "1", "true", "yes"):
            os.environ["GGML_CUDA_ENABLE_UNIFIED_MEMORY"] = "1"
            print("Unified Memory: ENABLED (via GGML_CUDA_ENABLE_UNIFIED_MEMORY=1)")
        else:
            os.environ.pop("GGML_CUDA_ENABLE_UNIFIED_MEMORY", None)
            print("Unified Memory: DISABLED")
        
        defrag_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "defrag_vram.py")
        if os.path.exists(defrag_script):
            try:
                subprocess.run(["python3", defrag_script], check=False)
            except Exception as e:
                print(f"Failed to sweep VRAM: {e}")
                
        bin_path = os.path.expanduser(cfg["BIN"])
        cmd = [bin_path]

        # Build command args list using unified builder
        cmd.extend(build_server_args(cfg, app.enabled_fields))
        
        print(f"Executing: {' '.join(shlex.quote(arg) for arg in cmd)}")
        print("-" * 60)
        os.execvp(bin_path, cmd)