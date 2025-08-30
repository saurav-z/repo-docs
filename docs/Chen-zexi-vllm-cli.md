# vLLM CLI: Supercharge Your LLMs with a Powerful Command-Line Interface

[![CI](https://github.com/Chen-zexi/vllm-cli/actions/workflows/ci.yml/badge.svg)](https://github.com/Chen-zexi/vllm-cli/actions/workflows/ci.yml)
[![Release](https://github.com/Chen-zexi/vllm-cli/actions/workflows/python-publish.yml/badge.svg)](https://github.com/Chen-zexi/vllm-cli/actions/workflows/python-publish.yml)
[![PyPI version](https://badge.fury.io/py/vllm-cli.svg)](https://badge.fury.io/py/vllm-cli)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyPI Downloads](https://static.pepy.tech/badge/vllm-cli)](https://pepy.tech/projects/vllm-cli)

**vLLM CLI provides a user-friendly command-line interface to serve and manage Large Language Models (LLMs) with ease, leveraging the speed and efficiency of vLLM.**

Enhance your LLM workflow with these key features:

*   **Interactive Mode:** Navigate and manage your LLMs through a rich, menu-driven terminal interface.
*   **Command-Line Mode:** Automate and script LLM operations directly from the command line.
*   **Model Management:** Seamlessly discover and load models from Hugging Face Hub and Ollama.
*   **Configuration Profiles:** Utilize pre-configured or custom server profiles tailored to your specific needs.
*   **Server Monitoring:** Keep tabs on your active vLLM servers in real-time.
*   **System Information:** Quickly assess your GPU, memory, and CUDA compatibility.
*   **Advanced Configuration:** Fine-tune vLLM parameters for optimal performance with validation.
*   **Multi-Model Proxy (Experimental):** Serve multiple LLMs through a single API endpoint (under active development).
*   **Hardware-Optimized Profiles:**  Pre-configured profiles for optimal serving of GPT-OSS models on various NVIDIA GPUs.

**[Explore the vLLM CLI Repository](https://github.com/Chen-zexi/vllm-cli)**

## What's New

### Multi-Model Proxy Server (Experimental)

*   Serve multiple models through a single API endpoint.
*   Dynamically add/remove models without service interruption.
*   Efficient GPU resource allocation with vLLM's sleep/wake functionality.
*   User-friendly setup wizard.

  Read more in the [🌐 Multi-Model Proxy Guide](docs/multi-model-proxy.md).

### Hardware-Optimized Profiles for GPT-OSS Models

*   **`gpt_oss_ampere`**: Optimized for NVIDIA A100 GPUs.
*   **`gpt_oss_hopper`**: Optimized for NVIDIA H100/H200 GPUs.
*   **`gpt_oss_blackwell`**: Optimized for NVIDIA Blackwell GPUs.

### Shortcuts System

*   Save and launch your favorite model + profile combinations with simple shortcuts:
    ```bash
    vllm-cli serve --shortcut my-gpt-server
    ```

### Ollama Integration

*   Automatic Ollama model discovery
*   GGUF format support (experimental)
*   System and user directory scanning

### Enhanced Configuration

*   Manage environment variables (universal and profile-specific).
*   Select specific GPUs for serving (`--device 0,1`).
*   Expanded system information, including vLLM feature detection.

See [CHANGELOG.md](CHANGELOG.md) for detailed release notes.

## Quick Start

### Installation

**Important:  vLLM and CUDA Compatibility:** Ensure your vLLM and PyTorch versions are compatible to avoid errors.

#### Option 1: Recommended - Install vLLM separately, then install vLLM CLI

```bash
# Install vLLM -- Skip if vLLM is installed in your environment
uv venv --python 3.12 --seed
source .venv/bin/activate
uv pip install vllm --torch-backend=auto
# Or specify a backend: uv pip install vllm --torch-backend=cu128

# Install vLLM CLI
uv pip install --upgrade vllm-cli
uv run vllm-cli

# If using conda:
# Activate the conda environment with vllm installed.
pip install vllm-cli
vllm-cli
```

#### Option 2: Install vLLM CLI + vLLM

```bash
pip install vllm-cli[vllm]
vllm-cli
```

#### Option 3: Build from Source (Requires vLLM installation)

```bash
git clone https://github.com/Chen-zexi/vllm-cli.git
cd vllm-cli
pip install -e .
```

#### Option 4: Isolated Installation (pipx/system packages)

```bash
# If you do not want to use virtual environment and want to install vLLM along with vLLM CLI
pipx install "vllm-cli[vllm]"

# If you want to install pre-release version
pipx install --pip-args="--pre" "vllm-cli[vllm]"
```

### Prerequisites

*   Python 3.9+
*   CUDA-compatible GPU (recommended)
*   vLLM package installed
*   For dependency issues, see [Troubleshooting Guide](docs/troubleshooting.md#dependency-conflicts)

### Basic Usage

```bash
# Interactive mode - menu-driven interface
vllm-cl
# Serve a model
vllm-cli serve --model openai/gpt-oss-20b

# Use a shortcut
vllm-cli serve --shortcut my-model
```

For detailed usage, see the [📘 Usage Guide](docs/usage-guide.md) and [🌐 Multi-Model Proxy Guide](docs/multi-model-proxy.md).

## Configuration

### Built-in Profiles

vLLM CLI offers 7 pre-configured profiles:

**General Purpose:**
*   `standard`
*   `high_throughput`
*   `low_memory`
*   `moe_optimized`

**Hardware-Specific (GPT-OSS):**
*   `gpt_oss_ampere`
*   `gpt_oss_hopper`
*   `gpt_oss_blackwell`

See the [📋 Profiles Guide](docs/profiles.md) for more information.

### Configuration Files

*   Main Config: `~/.config/vllm-cli/config.yaml`
*   User Profiles: `~/.config/vllm-cli/user_profiles.json`
*   Shortcuts: `~/.config/vllm-cli/shortcuts.json`

## Documentation

*   [📘 Usage Guide](docs/usage-guide.md)
*   [🌐 Multi-Model Proxy](docs/multi-model-proxy.md)
*   [📋 Profiles Guide](docs/profiles.md)
*   [❓ Troubleshooting](docs/troubleshooting.md)
*   [📸 Screenshots](docs/screenshots.md)
*   [🔍 Model Discovery](docs/MODEL_DISCOVERY_QUICK_REF.md)
*   [🦙 Ollama Integration](docs/ollama-integration.md)
*   [⚙️ Custom Models](docs/custom-model-serving.md)
*   [🗺️ Roadmap](docs/roadmap.md)

## Integration with hf-model-tool

vLLM CLI utilizes [hf-model-tool](https://github.com/Chen-zexi/hf-model-tool) for efficient model discovery.

## Development

### Project Structure

```
src/vllm_cli/
├── cli/           # CLI command handling
├── config/        # Configuration management
├── models/        # Model management
├── server/        # Server lifecycle
├── ui/            # Terminal interface
└── schemas/       # JSON schemas
```

### Contributing

Contributions are welcome! Please submit pull requests or open issues.

## License

MIT License - see [LICENSE](LICENSE) file.
```
Key improvements and SEO optimizations:

*   **Clear Hook:**  A concise, attention-grabbing opening sentence.
*   **Keyword Optimization:**  Uses relevant keywords like "vLLM," "LLMs," "command-line," "interface," and "model management" throughout the text.
*   **Structured Headings:**  Uses clear, descriptive headings (e.g., "Quick Start," "Configuration") for better readability and SEO.
*   **Bulleted Lists:**  Employs bullet points to highlight key features, making the information easy to scan.
*   **Concise Language:**  Rephrases text for greater clarity and impact.
*   **Internal Linking:** Includes links to other sections to improve the user experience.
*   **Focus on Benefits:**  Highlights the advantages of using vLLM CLI.
*   **Action-Oriented:**  Encourages users to explore the repository and features.
*   **Includes the Original Badges:** Added all the original badges to make the README self-contained.
*   **Expanded on Information:** Adds more details to the "What's New" section.
*   **Improved Quick Start Section:**  Offers clear installation steps.
*   **More Readable Format:** Increased readability by using better formatting (e.g., code blocks, lists).
*   **Focus on Value Proposition:** Emphasizes the core value of the project.
*   **Links Back:** Includes a link back to the original repo.