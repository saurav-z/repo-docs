# vLLM CLI: Command-Line Interface for Serving Large Language Models

**Effortlessly serve and manage your LLMs with vLLM using a user-friendly command-line interface.**  [(Back to original repo)](https://github.com/Chen-zexi/vllm-cli)

vLLM CLI provides a powerful and flexible command-line interface (CLI) for interacting with and managing Large Language Models (LLMs) using the vLLM framework.  Whether you're experimenting, automating, or building production systems, vLLM CLI offers a comprehensive suite of tools.

**Key Features:**

*   **Interactive Mode:** Navigate and manage your models with a rich, menu-driven terminal interface.
*   **Command-Line Mode:** Automate and script your LLM workflows with direct CLI commands.
*   **Model Management:** Discover and manage local models, with support for Hugging Face and Ollama.
*   **Configuration Profiles:** Utilize pre-configured profiles or create custom server configurations tailored to your needs.
*   **Server Monitoring:** Monitor active vLLM servers in real-time.
*   **System Information:** Get comprehensive information about your GPU, memory, and CUDA compatibility.
*   **Advanced Configuration:** Fine-tune vLLM parameters with validation for optimal performance.
*   **Multi-Model Proxy (Experimental):** Serve multiple LLMs through a single API endpoint for simplified access and management.
*   **Hardware Optimized Profiles:** Pre-configured profiles for optimal performance on specific NVIDIA GPU architectures (Ampere, Hopper, Blackwell).
*   **Ollama Integration:** Full support for Ollama models, including GGUF format (experimental) and system/user directory scanning.
*   **Shortcuts:** Quickly launch your favorite model + profile combinations.

**What's New:**

*   **Multi-Model Proxy (Experimental):**
    *   Single API endpoint for multiple models
    *   Live model management (add/remove without stopping)
    *   Dynamic GPU resource allocation.
    *   User-friendly setup wizard.
    *   For complete documentation, see the [🌐 Multi-Model Proxy Guide](docs/multi-model-proxy.md).
*   **Hardware-Optimized Profiles:** Pre-configured profiles for optimal performance on NVIDIA GPU architectures:
    *   `gpt_oss_ampere` (A100 GPUs)
    *   `gpt_oss_hopper` (H100/H200 GPUs)
    *   `gpt_oss_blackwell` (Blackwell GPUs)
*   **Shortcuts System:** Easily launch your favorite model and profile combinations with a single command.
    ```bash
    vllm-cli serve --shortcut my-gpt-server
    ```
*   **Full Ollama Integration:**
    *   Automatic Ollama model discovery
    *   GGUF format support (experimental)
    *   System and user directory scanning
*   **Enhanced Configuration:**
    *   Environment variable management (universal and profile-specific)
    *   GPU selection (`--device 0,1`)
    *   Enhanced system information

**Quick Start**

**Installation**

*   **Recommended (Separate vLLM Installation):**
    ```bash
    # Install vLLM (if not already installed)
    uv venv --python 3.12 --seed  # Or your preferred Python version
    source .venv/bin/activate
    uv pip install vllm --torch-backend=auto
    # Install vLLM CLI
    uv pip install --upgrade vllm-cli
    uv run vllm-cli
    ```
    *  If you are using conda, you can activate the environment you have vllm installed in.

*   **Alternative (Install vLLM CLI + vLLM):**
    ```bash
    pip install vllm-cli[vllm]
    vllm-cli
    ```
*   **Build from Source:**
    ```bash
    git clone https://github.com/Chen-zexi/vllm-cli.git
    cd vllm-cli
    pip install -e .
    ```
*   **Isolated Installation (pipx/system packages):** Consider using uv or conda for better PyTorch/CUDA compatibility

    ```bash
    pipx install "vllm-cli[vllm]" # Installs vLLM CLI and vLLM
    ```

**Prerequisites**

*   Python 3.9+
*   CUDA-compatible GPU (recommended)
*   vLLM package installed (See installation instructions above).

**Basic Usage**

```bash
# Interactive mode
vllm-cli

# Serve a model
vllm-cli serve --model openai/gpt-oss-20b

# Use a shortcut
vllm-cli serve --shortcut my-model
```

**Configuration**

*   **Built-in Profiles:**
    *   `standard`
    *   `high_throughput`
    *   `low_memory`
    *   `moe_optimized`
    *   `gpt_oss_ampere`
    *   `gpt_oss_hopper`
    *   `gpt_oss_blackwell`
*   **Configuration Files:**
    *   Main Config: `~/.config/vllm-cli/config.yaml`
    *   User Profiles: `~/.config/vllm-cli/user_profiles.json`
    *   Shortcuts: `~/.config/vllm-cli/shortcuts.json`
*   See [**📋 Profiles Guide**](docs/profiles.md) for details.

**Documentation**

*   [**📘 Usage Guide**](docs/usage-guide.md)
*   [**🌐 Multi-Model Proxy**](docs/multi-model-proxy.md)
*   [**📋 Profiles Guide**](docs/profiles.md)
*   [**❓ Troubleshooting**](docs/troubleshooting.md)
*   [**📸 Screenshots**](docs/screenshots.md)
*   [**🔍 Model Discovery**](docs/MODEL_DISCOVERY_QUICK_REF.md)
*   [**🦙 Ollama Integration**](docs/ollama-integration.md)
*   [**⚙️ Custom Models**](docs/custom-model-serving.md)
*   [**🗺️ Roadmap**](docs/roadmap.md)

**Integration with hf-model-tool**

vLLM CLI leverages [hf-model-tool](https://github.com/Chen-zexi/hf-model-tool) for efficient model discovery and configuration.

**Development**

*   Project Structure: Overview of the directory structure.
*   Contributing: Guidelines for contributing to the project.

**License**

MIT License - see [LICENSE](LICENSE) for details.