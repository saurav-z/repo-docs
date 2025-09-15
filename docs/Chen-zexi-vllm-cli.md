# vLLM CLI: Supercharge Your LLM Serving with a Powerful Command-Line Interface

[**Explore vLLM CLI on GitHub**](https://github.com/Chen-zexi/vllm-cli)

**vLLM CLI** is a robust command-line interface (CLI) designed for efficiently serving Large Language Models (LLMs) using vLLM, offering both interactive and automated modes for maximum flexibility. This project makes it easy to manage and deploy LLMs with intuitive features and advanced configuration options.

[![CI](https://github.com/Chen-zexi/vllm-cli/actions/workflows/ci.yml/badge.svg)](https://github.com/Chen-zexi/vllm-cli/actions/workflows/ci.yml)
[![Release](https://github.com/Chen-zexi/vllm-cli/actions/workflows/python-publish.yml/badge.svg)](https://github.com/Chen-zexi/vllm-cli/actions/workflows/python-publish.yml)
[![PyPI version](https://badge.fury.io/py/vllm-cli.svg)](https://badge.fury.io/py/vllm-cli)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyPI Downloads](https://static.pepy.tech/badge/vllm-cli)](https://pepy.tech/projects/vllm-cli)

## Key Features:

*   **Interactive Mode:** Navigate and manage LLMs through a feature-rich terminal interface.
*   **Command-Line Mode:** Automate your LLM workflows with powerful CLI commands.
*   **Model Management:** Seamlessly discover and manage local models, including support for Hugging Face and Ollama.
*   **Configuration Profiles:** Utilize pre-configured or custom server profiles tailored to diverse use cases.
*   **Server Monitoring:** Monitor active vLLM servers in real-time, including GPU and system status.
*   **System Information:** Get instant insights into your system's hardware, including GPU, memory, and CUDA compatibility.
*   **Advanced Configuration:** Full control over vLLM parameters, ensuring optimal performance and customization.
*   **Multi-Model Proxy (Experimental):** Serve multiple LLMs through a single API endpoint for unified access and dynamic management.

## What's New?

*   **Hardware-Optimized Profiles:** Serving GPT-OSS Models optimized for NVIDIA GPU architectures like Ampere, Hopper, and Blackwell.
*   **Shortcuts System:** Quickly launch your favorite model + profile combinations with shortcuts.
*   **Full Ollama Integration:** Automatic discovery of Ollama models, GGUF format support, and system/user directory scanning.

## Quick Start

### Installation (Recommended: Install vLLM Separately)

1.  **Install vLLM** (if not already installed):
    ```bash
    uv venv --python 3.12 --seed
    source .venv/bin/activate
    uv pip install vllm --torch-backend=auto
    ```
    Or for a specific backend:
    ```bash
    uv pip install vllm --torch-backend=cu128
    ```
2.  **Install vLLM CLI:**
    ```bash
    uv pip install --upgrade vllm-cli
    uv run vllm-cli
    ```

    If using conda:
    ```bash
    # Activate your vllm environment
    pip install vllm-cli
    vllm-cli
    ```

### Installation Option 2: Install vLLM CLI + vLLM

```bash
pip install vllm-cli[vllm]
vllm-cli
```

### Installation Option 3: Build From Source (Still Requires Separate vLLM Install)

```bash
git clone https://github.com/Chen-zexi/vllm-cli.git
cd vllm-cli
pip install -e .
```

### Installation Option 4: For Isolated Installation (pipx/system packages)

```bash
# If you do not want to use virtual environment and want to install vLLM along with vLLM CLI
pipx install "vllm-cli[vllm]"

# If you want to install pre-release version
pipx install --pip-args="--pre" "vllm-cli[vllm]"
```

### Prerequisites
- Python 3.9+
- CUDA-compatible GPU (recommended)
- vLLM package installed
- For dependency issues, see [Troubleshooting Guide](docs/troubleshooting.md#dependency-conflicts)

### Basic Usage

```bash
# Interactive mode
vllm-cli
# Serve a model
vllm-cli serve --model openai/gpt-oss-20b

# Use a shortcut
vllm-cli serve --shortcut my-model
```

## Configuration

### Built-in Profiles

vLLM CLI includes 7 pre-configured profiles optimized for different use cases:

**General Purpose:**

*   `standard` - Minimal configuration with smart defaults
*   `high_throughput` - Maximum performance configuration
*   `low_memory` - Memory-constrained environments
*   `moe_optimized` - Optimized for Mixture of Experts models

**Hardware-Specific (GPT-OSS):**

*   `gpt_oss_ampere` - NVIDIA A100 GPUs
*   `gpt_oss_hopper` - NVIDIA H100/H200 GPUs
*   `gpt_oss_blackwell` - NVIDIA Blackwell GPUs

See [**Profiles Guide**](docs/profiles.md) for detailed information.

### Configuration Files
- **Main Config**: `~/.config/vllm-cli/config.yaml`
- **User Profiles**: `~/.config/vllm-cli/user_profiles.json`
- **Shortcuts**: `~/.config/vllm-cli/shortcuts.json`

## Documentation

*   [**Usage Guide**](docs/usage-guide.md): Comprehensive usage instructions.
*   [**Multi-Model Proxy**](docs/multi-model-proxy.md): Serve multiple models simultaneously.
*   [**Profiles Guide**](docs/profiles.md): Details on built-in profiles.
*   [**Troubleshooting**](docs/troubleshooting.md): Common issues and solutions.
*   [**Screenshots**](docs/screenshots.md): Visual overview.
*   [**Model Discovery**](docs/MODEL_DISCOVERY_QUICK_REF.md): Model management guide.
*   [**Ollama Integration**](docs/ollama-integration.md): Using Ollama models.
*   [**Custom Models**](docs/custom-model-serving.md): Serving custom models.
*   [**Roadmap**](docs/roadmap.md): Future development plans.

## Integration with hf-model-tool

vLLM CLI integrates with [hf-model-tool](https://github.com/Chen-zexi/hf-model-tool) for model discovery:

*   Comprehensive model scanning.
*   Ollama model support.
*   Shared configuration.

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

Contributions are welcome! Please feel free to open an issue or submit a pull request.

## License

MIT License - see [LICENSE](LICENSE) file for details.
```
Key changes and improvements:

*   **SEO-Optimized Title and Introduction:** Uses keywords like "vLLM CLI," "LLM serving," "command-line interface," and "Large Language Models" in the title and introduction.  The introductory sentence is rewritten to be more engaging and hook the reader.
*   **Concise Feature Listing:**  Uses bullet points for easy scanning and emphasizes key benefits.
*   **Clear Headings:** Organizes the README with clear, descriptive headings.
*   **Quick Start Emphasis:**  Highlights the Quick Start section for easy onboarding.  Installation is streamlined.
*   **Added Emphasis on Experimental Features** Highlighted that multi model proxy is an experimental feature.
*   **Documentation Links:**  Provides direct links to important documentation sections.
*   **Project Structure:** Includes the project structure for developers.
*   **Call to Action:** Encourages contributions.
*   **Link Back to Repo:** Ensures the primary repository is easily accessible.