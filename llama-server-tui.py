#!/usr/bin/env python3
import os
import json
import shlex
from textual.app import App, ComposeResult
from textual.screen import ModalScreen
from textual.containers import VerticalScroll, Horizontal, Vertical, Container
from textual.widgets import Footer, Input, Button, Label, Collapsible, Checkbox, Static, Select
from textual.reactive import reactive
from pathlib import Path

CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(CONFIG_DIR, "config.json")

ASCII_HEADER = """  ╦  ╦  ╔═╗ ╔╦╗ ╔═╗   ╦  ╔═╗ ╦ ╦ ╔╗╔ ╔═╗ ╦ ╦ ╔═╗ ╦═╗
  ║  ║  ╠═╣ ║║║ ╠═╣   ║  ╠═╣ ║ ║ ║║║ ║   ╠═╣ ║╣  ╠╦╝
  ╩═╝╩═╝╩ ╩ ╩ ╩ ╩ ╩   ╩═╝╩ ╩ ╚═╝ ╝╚╝ ╚═╝ ╩ ╩ ╚═╝ ╩╚═"""

DEFAULT_CONFIG = {
    "BIN": os.path.expanduser("~/llama.cpp/build/bin/llama-server"),
    "MODEL": "/home/peter/.cache/huggingface/hub/models--unsloth--Qwen3.6-27B-MTP-GGUF/snapshots/84362f6d157c935ba13228710689cce2922e408a/Qwen3.6-27B-UD-Q4_K_XL.gguf",
    "HOST": "10.0.0.2",
    "PORT": "8000",
    "API_KEY": "",
    "ALIAS": "qwen3.6-27b-mtp",
    "NGL": "99",
    "CTX": "150000",
    "NP": "1",
    "THREADS": "-1",
    "THREADS_BATCH": "8",
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
    "SPEC_MIN": "0.05",
    "SPEC_DRAFT_N_MIN": "0",
    "JINJA": "on",
    "REASONING": "auto",
    "REASONING_FORMAT": "auto",
    "REASONING_BUDGET": "-1",
    "METRICS": "off",
    "SEED": "-1",
    "CHAT_TEMPLATE": "",
    "CHAT_TEMPLATE_FILE": "",
    "TEMPLATE_KWARGS": "",
    "MMPROJ": "",
    "MMPROJ_OFFLOAD": "on",
    "SKIP_CHAT_PARSING": "off",
    "WEBUI": "auto",
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
    "DRY_MULTIPLIER": "0.8",
    "DRY_BASE": "1.75",
    "DRY_ALLOWED_LENGTH": "2",
    "DRY_PENALTY_LAST_N": "-1",
    "XTC_PROBABILITY": "0.5",
    "XTC_THRESHOLD": "0.1",
    "EXTRA_FLAGS": "",
}

PARAM_HELP = {
    "BIN": "Path to llama-server binary executable.",
    "MODEL": "Path to the GGUF model file to load.",
    "HOST": "IP address the server listens on. Use 0.0.0.0 for all interfaces.",
    "PORT": "TCP port number for the HTTP server.",
    "API_KEY": "API key for authentication. Comma-separated for multiple keys. Leave empty to disable.",
    "ALIAS": "Model name alias returned in API responses (e.g. /v1/models).",
    "NGL": "Number of layers offloaded to GPU. Use 99 for full offload.",
    "CTX": "Context window size in tokens. Qwen3.6 supports up to 262K natively. Larger uses more VRAM.",
    "NP": "Number of parallel request slots (concurrent users).",
    "THREADS": "CPU threads for generation. -1 = auto. Set to physical core count for best perf.",
    "THREADS_BATCH": "CPU threads for batch prompt evaluation. -1 = auto. Set to physical core count.",
    "BATCH_SIZE": "Logical batch size for prompt evaluation. Larger = faster prefill, more VRAM.",
    "UBATCH_SIZE": "Physical GPU batch size. Smaller = less VRAM, larger = faster. Max = batch size.",
    "FLASH_ATTN": "Flash Attention: on/off/auto. Always enable for Qwen3.6 — reduces VRAM significantly.",
    "CACHE_K": "Quantization for KV cache keys. q8_0 halves memory vs f16 with minimal quality loss.",
    "CACHE_V": "Quantization for KV cache values. q8_0 recommended. Use bf16 if output seems garbled.",
    "MLOCK": "Lock model in RAM to prevent OS swapping (on/off). Requires sufficient RAM.",
    "TEMP": "Sampling temperature. Qwen3.6: 0.6 for coding/thinking, 0.7 for non-thinking, 1.0 for general thinking.",
    "TOP_P": "Nucleus sampling. Qwen3.6: 0.95 for thinking mode, 0.8 for non-thinking.",
    "TOP_K": "Only sample from top K tokens. Qwen3.6 recommended: 20.",
    "MIN_P": "Minimum probability threshold relative to top token.",
    "PRESENCE_PENALTY": "Penalize tokens already present. Qwen3.6: 0.0 for thinking, 1.5 for non-thinking.",
    "REPEAT_PENALTY": "Penalize repeated tokens. 1.0 = no penalty. Keep at 1.0 for Qwen3.6.",
    "SPEC_TYPE": "Speculative decoding type. Use draft-mtp for Qwen3.6-MTP (native multi-token prediction).",
    "SPEC_MAX": "Max draft tokens per speculation step. 2-3 optimal for Qwen3.6-MTP. Experiment 1-6.",
    "SPEC_MIN": "Min acceptance probability for draft tokens. Lower = more aggressive speculation.",
    "SPEC_DRAFT_N_MIN": "Min draft tokens per step. >0 avoids wasteful short speculations.",
    "JINJA": "Enable Jinja2 chat template rendering. Required for tool/function calling.",
    "REASONING": "Qwen3.6 thinking mode: on (always think), off (no thinking), auto (model decides).",
    "REASONING_FORMAT": "How <think> blocks are returned: auto, deepseek, none. Use deepseek for OpenAI-compatible clients.",
    "REASONING_BUDGET": "Token budget for thinking. -1 = unlimited, 0 = disabled, N = max thinking tokens.",
    "METRICS": "Enable Prometheus metrics at /metrics endpoint. Tracks tokens/s, slots, latency.",
    "SEED": "RNG seed for reproducible output. -1 = random.",
    "CHAT_TEMPLATE": "Force a built-in static chat template format (e.g. chatml, llama3, gemma). Typically empty when using --jinja.",
    "CHAT_TEMPLATE_FILE": "Path to a custom Jinja chat template file (.jinja or .txt). Override embedded templates.",
    "TEMPLATE_KWARGS": "JSON string for chat template arguments (e.g. {\"preserve_thinking\":true}). Crucial to retain <think> tags.",
    "MMPROJ": "Path to a multimodal projector file (companion GGUF for vision models like Qwen2-VL or Llama 3.2 Vision).",
    "MMPROJ_OFFLOAD": "Offload visual projector computation to GPU (on/off). Significantly improves visual prefill latency.",
    "SKIP_CHAT_PARSING": "Force a pure content parser (on/off). Everything, including reasoning and tool calls, is printed directly inside content.",
    "WEBUI": "Enable and configure the built-in server web interface (auto/on/off).",
    "SPLIT_MODE": "How to split model across multiple GPUs (none/row/layer). Use 'none' for single GPU.",
    "MAIN_GPU": "The primary GPU index (0, 1, etc.) used for coordinating model tensors.",
    "PREDICT": "Max number of tokens to predict/generate. Set to -1 for infinite generation.",
    "CTX_CHECKPOINTS": "Number of context checkpoints to store. Helps reduce VRAM when running huge contexts.",
    "CACHE_RAM": "GPU memory (in MB) allocated to the KV cache structure. Helps reserve VRAM safely.",
    "CONT_BATCHING": "Enable continuous batching of multiple request sequences (on/off).",
    "DRAFT_MAX": "Max speculative draft tokens to generate per step (e.g., 5). Leave empty to use model default.",
    "DRAFT_P_MIN": "Min speculative draft token acceptance probability (e.g., 0.50).",
    "IMAGE_MIN_TOKENS": "Minimum token budget allocated for an input image (e.g., 1024).",
    "IMAGE_MAX_TOKENS": "Maximum token budget allocated for an input image (e.g., 4096).",
    "TOOLS": "Enable built-in tools for AI agents (all/none/list). Pre-enabled as all.",
    "TENSOR_SPLIT": "Fraction of model to allocate to each GPU (comma-separated, e.g. 3,1). Empty for single GPU.",
    "NUMA": "NUMA optimization style to use. Recommended: none for single-socket motherboards.",
    "NO_MMAP": "Disable memory-mapping completely to load all weights directly into CPU/GPU RAM (on/off).",
    "TIMEOUT": "Server connection timeout in seconds. 1800 recommended for large prompts.",
    "THREADS_HTTP": "Number of threads to process HTTP requests. -1 maps automatically to physical core count.",
    "CACHE_PROMPT": "Enable prompt caching to reuse past context in multi-turn dialogues (on/off).",
    "CACHE_REUSE": "Minimum number of prompt tokens to reuse from cache. 256 optimal.",
    "DRY_MULTIPLIER": "DRY sampler multiplier. 0.8 is optimal for Qwen 3.6 to completely prevent repetition loops.",
    "DRY_BASE": "DRY sampler base penalty exponent (default: 1.75).",
    "DRY_ALLOWED_LENGTH": "Number of exact matching tokens allowed before DRY penalty is applied (default: 2).",
    "DRY_PENALTY_LAST_N": "Apply DRY penalty to the last N tokens only. -1 applies to the entire context.",
    "XTC_PROBABILITY": "XTC sampler probability. 0.5 balances vocabulary diversity and logical consistency perfectly.",
    "XTC_THRESHOLD": "XTC sampler minimum probability threshold. 0.1 excludes standard obvious choices.",
    "EXTRA_FLAGS": "Any additional custom command-line arguments to pass directly to llama-server (e.g. --verbose --grp-attn-n 4).",
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
}

class ParameterField(Horizontal):
    enabled = reactive(True)

    def __init__(self, key: str, label: str, value: str, is_mandatory: bool = False, is_enabled: bool = True):
        super().__init__()
        self.key = key
        self.label_text = label
        self.initial_value = value
        self.is_mandatory = is_mandatory
        self._initial_enabled = is_enabled

    @property
    def value(self) -> str:
        if self.key in SELECT_OPTIONS:
            val = self.query_one(Select).value
            return str(val) if val is not None and val != Select.BLANK else ""
        return self.query_one(Input).value

    @value.setter
    def value(self, val: str) -> None:
        if self.key in SELECT_OPTIONS:
            self.query_one(Select).value = val
        else:
            self.query_one(Input).value = val

    def compose(self) -> ComposeResult:
        if not self.is_mandatory:
            yield Label(" ✔ ", classes="custom-checkbox", id=f"chk-{self.key}")
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

    def watch_enabled(self, enabled: bool) -> None:
        self.update_visuals()

    def update_visuals(self) -> None:
        if not self.is_mounted:
            return
        if not self.is_mandatory:
            try:
                chk = self.query_one(f"#chk-{self.key}", Label)
                chk.update(" ✔ " if self.enabled else " ✖ ")
            except Exception:
                pass
            if self.enabled:
                self.remove_class("dimmed")
            else:
                self.add_class("dimmed")
            try:
                if self.key in SELECT_OPTIONS:
                    self.query_one(Select).disabled = not self.enabled
                else:
                    self.query_one(Input).disabled = not self.enabled
            except Exception:
                pass

    def on_mount(self) -> None:
        self.enabled = self._initial_enabled
        self.update_visuals()

    def on_click(self, event) -> None:
        if not self.is_mandatory and hasattr(event, "control") and event.control and event.control.id == f"chk-{self.key}":
            self.enabled = not self.enabled



class FileBrowserModal(ModalScreen[str]):
    """Modal file browser for selecting model files."""

    BINDINGS = [("escape", "cancel", "Cancel")]

    DEFAULT_CSS = """
    FileBrowserModal {
        align: center middle;
        background: #000000 60%;
    }
    #file-browser-container {
        width: 90;
        height: 30;
        background: #504945;
        border: thick #fabd2f;
        padding: 1 2;
    }
    #fb-title {
        text-align: center;
        color: #fabd2f;
        text-style: bold;
        background: #504945;
        width: 100%;
        height: 1;
        margin-bottom: 1;
    }
    #fb-path {
        color: #83a598;
        background: #504945;
        width: 100%;
        height: 1;
        margin-bottom: 1;
    }
    #fb-file-list {
        height: 1fr;
        background: #3c3836;
        padding: 0 1;
    }
    .fb-entry {
        width: 100%;
        height: 1;
        background: #3c3836;
        color: #ebdbb2;
        padding: 0 1;
    }
    .fb-entry:hover {
        background: #504945;
    }
    .fb-entry-dir {
        color: #fabd2f;
    }
    .fb-entry-file {
        color: #b8bb26;
    }
    .fb-entry-parent {
        color: #fe8019;
        text-style: bold;
    }
    #fb-buttons {
        height: auto;
        align: center middle;
        margin-top: 1;
        background: #504945;
    }
    #fb-buttons Button {
        margin: 0 1;
        min-width: 12;
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

    def compose(self) -> ComposeResult:
        with Container(id="file-browser-container"):
            title = "📂 Select File" if self.file_extensions is None else "📂 Select Model File"
            yield Static(title, id="fb-title")
            yield Static(str(self.current_dir), id="fb-path")
            with VerticalScroll(id="fb-file-list"):
                yield from self._build_entries()
            with Horizontal(id="fb-buttons"):
                yield Button("Cancel", id="btn-fb-cancel")

    def _build_entries(self):
        """Build file/dir labels for current directory."""
        # Parent dir entry
        lbl = Label("📁 ..", classes="fb-entry fb-entry-parent")
        lbl.fb_path = str(self.current_dir.parent)
        lbl.fb_type = "parent"
        yield lbl
        try:
            for item in sorted(self.current_dir.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())):
                if item.name.startswith("."):
                    continue
                if item.is_dir():
                    lbl = Label(f"📁 {item.name}", classes="fb-entry fb-entry-dir")
                    lbl.fb_path = str(item)
                    lbl.fb_type = "dir"
                    yield lbl
                elif item.is_file():
                    if self.file_extensions is not None and item.suffix.lower() not in self.file_extensions:
                        continue
                    size_mb = item.stat().st_size / (1024 * 1024)
                    size_str = f"{size_mb:.0f}MB" if size_mb < 1024 else f"{size_mb/1024:.1f}GB"
                    lbl = Label(f"📄 {item.name}  ({size_str})", classes="fb-entry fb-entry-file")
                    lbl.fb_path = str(item)
                    lbl.fb_type = "file"
                    yield lbl
        except PermissionError:
            yield Label("⚠ Permission denied", classes="fb-entry")

    def _refresh_list(self):
        """Refresh the file list for current directory."""
        self.query_one("#fb-path", Static).update(str(self.current_dir))
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


class SaveProfileModal(ModalScreen[str]):
    """Modal input dialog for saving a profile as a name."""

    BINDINGS = [("escape", "cancel", "Cancel")]

    DEFAULT_CSS = """
    SaveProfileModal {
        align: center middle;
        background: #000000 60%;
    }
    #save-profile-container {
        width: 50;
        height: 13;
        background: #3c3836;
        border: thick #fabd2f;
        padding: 1 2;
        align: center middle;
    }
    #sp-title {
        color: #fabd2f;
        text-style: bold;
        text-align: center;
        width: 100%;
        margin-bottom: 1;
        background: #3c3836;
    }
    #sp-label {
        color: #ebdbb2;
        width: 100%;
        margin-bottom: 1;
        background: #3c3836;
    }
    #sp-input {
        width: 100%;
        background: #504945;
        border: none;
        color: #ebdbb2;
        padding: 0 1;
        height: 1;
        margin-bottom: 1;
    }
    #sp-buttons {
        align: center middle;
        height: auto;
        margin-top: 1;
        background: #3c3836;
    }
    #sp-buttons Button {
        margin: 0 1;
        min-width: 12;
    }
    #btn-sp-cancel {
        background: #665c54;
        color: #ebdbb2;
    }
    #btn-sp-save {
        background: #fabd2f;
        color: #282828;
        text-style: bold;
    }
    """

    def __init__(self, current_name: str = "default"):
        super().__init__()
        self.current_name = current_name

    def compose(self) -> ComposeResult:
        with Container(id="save-profile-container"):
            yield Static("💾 Save Profile As", id="sp-title")
            yield Label("Enter profile name:", id="sp-label")
            yield Input(value=self.current_name, id="sp-input")
            with Horizontal(id="sp-buttons"):
                yield Button("Cancel", id="btn-sp-cancel")
                yield Button("Save", id="btn-sp-save")

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


class LoadProfileModal(ModalScreen[str]):
    """Modal dialog to list and select profiles."""

    BINDINGS = [("escape", "cancel", "Cancel")]

    DEFAULT_CSS = """
    LoadProfileModal {
        align: center middle;
        background: #000000 60%;
    }
    #load-profile-container {
        width: 50;
        height: 20;
        background: #3c3836;
        border: thick #fabd2f;
        padding: 1 2;
        align: center middle;
    }
    #lp-title {
        color: #fabd2f;
        text-style: bold;
        text-align: center;
        width: 100%;
        margin-bottom: 1;
        background: #3c3836;
    }
    #lp-list {
        height: 1fr;
        background: #282828;
        padding: 0 1;
        width: 100%;
        margin-bottom: 1;
    }
    .lp-entry {
        width: 100%;
        height: 1;
        background: #282828;
        color: #ebdbb2;
        padding: 0 1;
    }
    .lp-entry:hover {
        background: #504945;
        color: #fabd2f;
        text-style: bold;
    }
    #lp-buttons {
        align: center middle;
        height: auto;
        background: #3c3836;
    }
    #lp-buttons Button {
        margin: 0 1;
        min-width: 12;
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
                yield Button("Cancel", id="btn-lp-cancel")

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


class AlertModal(ModalScreen[None]):
    """A beautiful centered modal dialog for alerts and notifications."""

    BINDINGS = [
        ("escape", "dismiss_modal", "Dismiss"),
        ("enter", "dismiss_modal", "Dismiss"),
        ("space", "dismiss_modal", "Dismiss"),
    ]

    DEFAULT_CSS = """
    AlertModal {
        align: center middle;
        background: #000000 60%;
    }
    #alert-container {
        width: 50;
        height: 11;
        background: #3c3836;
        border: thick #fabd2f;
        padding: 1 2;
        align: center middle;
    }
    #alert-title {
        color: #fabd2f;
        text-style: bold;
        text-align: center;
        width: 100%;
        margin-bottom: 1;
        background: #3c3836;
    }
    #alert-message {
        color: #ebdbb2;
        text-align: center;
        width: 100%;
        margin-bottom: 1;
        background: #3c3836;
    }
    #alert-buttons {
        align: center middle;
        height: auto;
        background: #3c3836;
    }
    #btn-alert-ok {
        background: #fabd2f;
        color: #282828;
        text-style: bold;
        min-width: 12;
        align: center middle;
    }
    #btn-alert-ok:hover {
        background: #fe8019;
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
                yield Button("OK", id="btn-alert-ok")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-alert-ok":
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
    $surface: #282828;
    $boost: #282828;
    $panel: #282828;
    $background: #282828;

    Screen {
        background: #282828;
        padding: 0;
        margin: 0;
        border: none;
    }

    #ascii-header {
        dock: top;
        width: 100%;
        height: auto;
        background: #1d2021;
        color: #fabd2f;
        text-align: center;
        padding: 1 0;
    }
    #notification-banner {
        dock: top;
        width: 100%;
        height: 1;
        background: #3c3836;
        color: #fabd2f;
        text-align: center;
        text-style: bold;
        display: none;
    }
    #notification-banner.success {
        color: #b8bb26;
    }
    #notification-banner.error {
        color: #fb4934;
    }
    #main-scroll {
        padding: 1 2;
        background: $surface;
    }
    Input:focus {
        background: #fabd2f;
        color: $surface;
    }
    Select:focus SelectCurrent, SelectCurrent:focus {
        background: #fabd2f;
        color: $surface;
    }
    Label {
        color: #ebdbb2;
        text-style: bold;
    }
    .custom-checkbox {
        background: $surface;
        color: #fabd2f;
        text-style: bold;
        width: 3;
        margin-right: 1;
        content-align: center middle;
    }
    .custom-checkbox:hover {
        color: #fbf1c7;
    }
    Button {
        margin: 2 2;
        background: #3c3836;
        color: #ebdbb2;
        border: none;
    }
    Button:hover {
        background: #fe8019;
        color: $surface;
    }
    #btn-start {
        background: #fabd2f;
        color: $surface;
        text-style: bold;
    }
    #btn-start:hover {
        background: #fe8019;
    }
    #buttons {
        dock: bottom;
        align: center middle;
        height: 4;
        background: $surface;
        border-top: solid #3c3836;
    }
    #buttons Button {
        margin: 0 2;
        height: 1;
        min-height: 0;
    }
    .grid-container {
        layout: grid;
        grid-size: 2;
        height: auto;
        padding: 0;
        grid-columns: 1fr 1fr;
        grid-rows: auto;
        grid-gutter: 1 6;
        background: $surface;
    }
    ParameterField {
        layout: horizontal;
        height: 1;
        align: left middle;
        background: $surface;
    }
    .field-label {
        width: 25;
        background: $surface;
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
        width: 1fr;
        height: 1;
        border: none;
        background: transparent;
        padding: 0;
        margin: 0;
    }
    SelectCurrent {
        background: #3c3836;
        color: #ebdbb2;
        border: none;
        height: 1;
        padding: 0 1;
    }
    .spacer-checkbox {
        width: 3;
        margin-right: 1;
        background: $surface;
    }
    Collapsible {
        margin-right: 2;
    }
    Collapsible, CollapsibleTitle, Contents {
        background: $surface;
    }
    Collapsible:focus, CollapsibleTitle:focus, CollapsibleTitle:hover, Collapsible:focus-within, Collapsible.-expanded {
        background: $surface;
        background-tint: transparent;
    }
    Collapsible > Contents {
        background: $surface;
    }
    CollapsibleTitle {
        margin: 0 0 1 0;
        padding: 0;
        height: auto;
        color: #fabd2f;
        text-style: bold;
    }
    Contents {
        padding: 0;
        margin: 0;
        height: auto;
    }
    .dimmed {
        opacity: 0.3;
    }
    .info-icon {
        width: 3;
        color: #504945;
        background: transparent;
        text-style: bold;
        content-align: center middle;
    }
    .info-icon:hover {
        color: #928374;
    }
    .browse-btn {
        min-width: 4;
        width: 4;
        height: 1;
        margin: 0 0 0 1;
        padding: 0;
        background: #3c3836;
        color: #fabd2f;
        border: none;
    }
    .browse-btn:hover {
        background: #fe8019;
        color: #282828;
    }
    Tooltip {
        background: #504945;
        color: #ebdbb2;
        border-right: wide #1d2021;
        border-bottom: wide #1d2021;
        padding: 1 2;
        max-width: 50;
    }
    Toast {
        background: #3c3836;
        color: #ebdbb2;
        border-left: thick #fabd2f;
        padding: 1 2;
        height: auto;
        min-height: 3;
    }
    Toast:hover {
        background: #504945;
    }
    Toast .toast--title {
        color: #fabd2f;
        text-style: bold;
    }
    ToastRack {
        dock: none;
        position: absolute;
        offset: 0 0;
        layer: _toast;
        align: center middle;
        width: 100%;
        height: 100%;
    }
    """

    def __init__(self):
        super().__init__()
        self.config = DEFAULT_CONFIG.copy()
        self.should_start = False
        self.enabled_fields = {}
        
        # Ensure profiles directory exists
        self.profiles_dir = os.path.join(CONFIG_DIR, "profiles")
        os.makedirs(self.profiles_dir, exist_ok=True)
        self.active_profile = "default"
        
        # Load initial config
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, "r") as f:
                    loaded = json.load(f)
                    self.config.update(loaded)
                    self.active_profile = loaded.get("ACTIVE_PROFILE", "default")
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
            self.enabled_fields[k] = k not in disabled

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
        try:
            banner = self.query_one("#notification-banner", Static)
            banner.update(message)
            banner.remove_class("success", "error")
            banner.add_class(severity)
            banner.styles.display = "block"
            
            if hasattr(self, "_notif_timer") and self._notif_timer:
                try:
                    self._notif_timer.stop()
                except Exception:
                    pass
            
            self._notif_timer = self.set_timer(2.0, self.hide_notification)
        except Exception:
            pass

    def hide_notification(self) -> None:
        try:
            banner = self.query_one("#notification-banner", Static)
            banner.styles.display = "none"
            self._notif_timer = None
        except Exception:
            pass

    def compose(self) -> ComposeResult:
        yield Static(ASCII_HEADER, id="ascii-header")
        yield Static("", id="notification-banner")
        
        sections = [
            ("Infrastructure", [
                ("BIN", "Server Binary:"),
                ("MODEL", "Model Path:"),
                ("HOST", "Host IP:"),
                ("PORT", "Port:"),
                ("API_KEY", "API Key:"),
                ("ALIAS", "Model Alias:"),
                ("WEBUI", "Web UI (auto/on/off):"),
            ]),
            ("Hardware & Performance", [
                ("NGL", "GPU Layers (-ngl):"),
                ("CTX", "Context (-c):"),
                ("NP", "Parallel (-np):"),
                ("THREADS", "CPU Threads (-t):"),
                ("THREADS_BATCH", "Batch Threads (-tb):"),
                ("BATCH_SIZE", "Batch Size (-b):"),
                ("UBATCH_SIZE", "Micro Batch (-ub):"),
                ("FLASH_ATTN", "Flash Attention:"),
                ("CACHE_K", "KV Cache K:"),
                ("CACHE_V", "KV Cache V:"),
                ("MLOCK", "MLock (on/off):"),
                ("SPLIT_MODE", "Split Mode (-sm):"),
                ("MAIN_GPU", "Main GPU (-mg):"),
                ("TENSOR_SPLIT", "Tensor Split (-ts):"),
                ("NUMA", "NUMA Optimization:"),
            ]),
            ("Context & Cache", [
                ("CTX_CHECKPOINTS", "Ctx Checkpoints:"),
                ("CACHE_RAM", "Cache RAM MB:"),
                ("CONT_BATCHING", "Cont Batching (on/off):"),
                ("CACHE_PROMPT", "Prompt Caching:"),
                ("CACHE_REUSE", "Cache Reuse Min:"),
                ("NO_MMAP", "Disable MMAP (no-mmap):"),
            ]),
            ("HTTP Server", [
                ("TIMEOUT", "Server Timeout (sec):"),
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
                ("SPEC_TYPE", "Spec Type:"),
                ("SPEC_MAX", "Spec Max Draft:"),
                ("SPEC_MIN", "Spec Min Prob:"),
                ("SPEC_DRAFT_N_MIN", "Spec Min Draft:"),
                ("DRAFT_MAX", "Draft Max Steps:"),
                ("DRAFT_P_MIN", "Draft Min Prob:"),
            ]),
            ("Reasoning & Agentic (Qwen3.6)", [
                ("REASONING", "Thinking Mode:"),
                ("REASONING_FORMAT", "Thinking Format:"),
                ("REASONING_BUDGET", "Thinking Budget:"),
                ("TOOLS", "Agentic Tools (--tools):"),
            ]),
            ("Features & Monitoring", [
                ("JINJA", "Jinja Templates:"),
                ("METRICS", "Metrics (on/off):"),
            ]),
            ("Multimodal & Templates", [
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
                with Collapsible(title=title, collapsed=False):
                    with Container(classes="grid-container"):
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

        with Horizontal(id="buttons"):
            yield Button("Load Profile", id="btn-load-profile", variant="default")
            yield Button("Save As...", id="btn-save-profile", variant="primary")
            yield Button("Start Server", id="btn-start", variant="success")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-load-profile":
            self.push_screen(
                LoadProfileModal(profiles_dir=self.profiles_dir),
                callback=self._on_profile_loaded
            )
        elif event.button.id == "btn-save-profile":
            self.push_screen(
                SaveProfileModal(current_name=self.active_profile),
                callback=self._on_profile_saved_as
            )
        elif event.button.id == "btn-start":
            self.save_config()
            self.should_start = True
            self.exit()
        elif event.button.id == "btn-browse-model":
            model_field = self.query_one("#MODEL", Input)
            self.push_screen(
                FileBrowserModal(start_path=model_field.value),
                callback=self._on_model_selected,
            )
        elif event.button.id == "btn-browse-bin":
            bin_field = self.query_one("#BIN", Input)
            self.push_screen(
                FileBrowserModal(start_path=bin_field.value, file_extensions=None),
                callback=self._on_bin_selected,
            )
        elif event.button.id == "btn-browse-mmproj":
            mmproj_field = self.query_one("#MMPROJ", Input)
            self.push_screen(
                FileBrowserModal(start_path=mmproj_field.value),
                callback=self._on_mmproj_selected,
            )
        elif event.button.id == "btn-browse-template":
            tmpl_field = self.query_one("#CHAT_TEMPLATE_FILE", Input)
            self.push_screen(
                FileBrowserModal(start_path=tmpl_field.value, file_extensions=(".jinja", ".txt", ".tmpl")),
                callback=self._on_template_selected,
            )

    def _on_model_selected(self, path: str) -> None:
        if path:
            self.query_one("#MODEL", Input).value = path

    def _on_bin_selected(self, path: str) -> None:
        if path:
            self.query_one("#BIN", Input).value = path

    def _on_mmproj_selected(self, path: str) -> None:
        if path:
            self.query_one("#MMPROJ", Input).value = path

    def _on_template_selected(self, path: str) -> None:
        if path:
            self.query_one("#CHAT_TEMPLATE_FILE", Input).value = path

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
                    self.show_notification(f"✨ Profile '{profile_name}' loaded successfully! ✨", severity="success")
                except Exception as e:
                    self.show_notification(f"⚠ Error loading profile: {e} ⚠", severity="error")

    def _on_profile_saved_as(self, profile_name: str) -> None:
        if profile_name:
            self.active_profile = profile_name
            self.save_config()  # Dual-saves to both config.json and the new profile
            self.show_notification(f"✨ Profile saved as '{profile_name}'! ✨", severity="success")

    def load_profile_data(self, data: dict) -> None:
        self.config.update(data)
        disabled = data.get("DISABLED_FIELDS", [])
        try:
            fields = list(self.query(ParameterField))
        except Exception:
            fields = []

        if fields:
            for field in fields:
                if field.key in data:
                    field.value = str(data[field.key])
                field.enabled = field.key not in disabled
        else:
            for k in DEFAULT_CONFIG:
                self.enabled_fields[k] = k not in disabled

    def save_config(self):
        disabled = []
        try:
            fields = list(self.query(ParameterField))
        except Exception:
            fields = []

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



if __name__ == "__main__":
    app = LlamaConfigApp()
    app.run()

    if app.should_start:
        os.system("clear")
        print("Starting Llama Server...")
        cfg = app.config
        
        bin_path = os.path.expanduser(cfg["BIN"])
        cmd = [bin_path]

        # Map UI keys to CLI flags (key=value style)
        flag_mapping = {
            "MODEL": "-m",
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

        for key, value in cfg.items():
            if app.enabled_fields.get(key, True) and key in flag_mapping:
                if not value.strip():
                    continue
                cmd.extend([flag_mapping[key], value.strip()])
        
        # Special handling for flags with non-standard formats
        for key, value in cfg.items():
            if app.enabled_fields.get(key, True):
                if key == "FLASH_ATTN":
                    cmd.extend(["--flash-attn", value])
                elif key == "JINJA":
                    if value.lower() in ("on", "1", "true", "yes"):
                        cmd.append("--jinja")
                    else:
                        cmd.append("--no-jinja")
                elif key == "MLOCK":
                    if value.lower() in ("on", "1", "true", "yes"):
                        cmd.append("--mlock")
                elif key == "METRICS":
                    if value.lower() in ("on", "1", "true", "yes"):
                        cmd.append("--metrics")
                elif key == "WEBUI":
                    if value.lower() in ("on", "1", "true", "yes", "auto"):
                        cmd.append("--webui")
                    else:
                        cmd.append("--no-webui")
                elif key == "CONT_BATCHING":
                    if value.lower() in ("on", "1", "true", "yes"):
                        cmd.append("--cont-batching")
                    else:
                        cmd.append("--no-cont-batching")
                elif key == "MERGE_QKV":
                    if value.lower() in ("on", "1", "true", "yes"):
                        cmd.append("--merge-qkv")
                elif key == "MERGE_EXPERTS":
                    if value.lower() in ("on", "1", "true", "yes"):
                        cmd.append("--merge-up-gate-experts")
                elif key == "API_KEY" and value.strip():
                    cmd.extend(["--api-key", value.strip()])
                elif key == "ALIAS" and value.strip():
                    cmd.extend(["-a", value.strip()])
                elif key == "MMPROJ_OFFLOAD":
                    if value.lower() in ("off", "0", "false", "no"):
                        cmd.append("--no-mmproj-offload")
                elif key == "SKIP_CHAT_PARSING":
                    if value.lower() in ("on", "1", "true", "yes"):
                        cmd.append("--skip-chat-parsing")
                elif key == "NUMA":
                    if value != "none":
                        cmd.extend(["--numa", value])
                elif key == "NO_MMAP":
                    if value == "on":
                        cmd.append("--no-mmap")
                elif key == "CACHE_PROMPT":
                    if value == "on":
                        cmd.append("--cache-prompt")
                    else:
                        cmd.append("--no-cache-prompt")
                elif key == "EXTRA_FLAGS" and value.strip():
                    cmd.extend(shlex.split(value.strip()))
        
        print(f"Executing: {' '.join(shlex.quote(arg) for arg in cmd)}")
        print("-" * 60)
        os.execvp(bin_path, cmd)