# llama-server-tui

A keyboard-driven terminal user interface (TUI) launcher for `llama-server`. It provides a clean dashboard to configure and run the server without copy-pasting long command-line arguments.

```
  ╦  ╦  ╔═╗ ╔╦╗ ╔═╗   ╦  ╔═╗ ╦ ╦ ╔╗╔ ╔═╗ ╦ ╦ ╔═╗ ╦═╗
  ║  ║  ╠═╣ ║║║ ╠═╣   ║  ╠═╣ ║ ║ ║║║ ║   ╠═╣ ║╣  ╠╦╝
  ╩═╝╩═╝╩ ╩ ╩ ╩ ╩ ╩   ╩═╝╩ ╩ ╚═╝ ╝╚╝ ╚═╝ ╩ ╩ ╚═╝ ╩╚═
```

## Key Highlights

- **Keyboard & Mouse Support**: Fast keyboard navigation, checkboxes, and inline directory selectors, with full mouse support for clicking, scrolling, and focus controls.
- **Off-Thread I/O**: Directory navigation and GGUF header parsing (block counts, architecture, KV channels) run in background worker threads, keeping the TUI responsive even when loading massive GGUF models.
- **Zero-Config Bootstrapping**: Automatically creates the profiles directory and sets up a standard default profile on the first run.
- **Standard Stack**: Written in pure Python and Textual.

## Installation and Quick Start

### 1. Install Prerequisites

The only dependency is Textual:

```bash
pip install textual
```

### 2. Run the Launcher

You do not need to manually create or edit any configuration files. The application self-bootstraps completely. Simply run the launcher:

```bash
python3 llama-server-tui.py
```

Alternatively, you can run the wrapper script:

```bash
./llama-server-start
```

On its first run, the launcher automatically:
1. Creates a `./profiles/` directory in the project folder.
2. Initializes a standard, generic `default.json` profile.
3. Opens the interface, where you can select your `llama-server` binary and GGUF model paths using the built-in directory browsers.

## Profiles & Configuration

Your custom profiles are saved as individual JSON files inside the `./profiles/` directory, while the active session configuration is saved in `config.json` in the root folder. Both are ignored by Git to keep your local paths and network settings completely private.

## License

MIT
