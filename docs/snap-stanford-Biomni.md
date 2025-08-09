# Biomni: Revolutionizing Biomedical Research with an AI Agent

**Biomni empowers scientists to accelerate biomedical research by autonomously executing complex tasks using cutting-edge AI.** [Learn more on the original repo](https://github.com/snap-stanford/Biomni).

<p align="center">
  <img src="./figs/biomni_logo.png" alt="Biomni Logo" width="600px" />
</p>

<p align="center">
<a href="https://join.slack.com/t/biomnigroup/shared_invite/zt-38dat07mc-mmDIYzyCrNtV4atULTHRiw">
<img src="https://img.shields.io/badge/Join-Slack-4A154B?style=for-the-badge&logo=slack" alt="Join Slack" />
</a>
<a href="https://biomni.stanford.edu">
<img src="https://img.shields.io/badge/Try-Web%20UI-blue?style=for-the-badge" alt="Web UI" />
</a>
<a href="https://x.com/ProjectBiomni">
<img src="https://img.shields.io/badge/Follow-on%20X-black?style=for-the-badge&logo=x" alt="Follow on X" />
</a>
<a href="https://www.linkedin.com/company/project-biomni">
<img src="https://img.shields.io/badge/Follow-LinkedIn-0077B5?style=for-the-badge&logo=linkedin" alt="Follow on LinkedIn" />
</a>
<a href="https://www.biorxiv.org/content/10.1101/2025.05.30.656746v1">
<img src="https://img.shields.io/badge/Read-Paper-green?style=for-the-badge" alt="Paper" />
</a>
</p>


## Key Features

*   **Autonomous Task Execution:** Biomni utilizes advanced AI to autonomously plan and execute a wide array of biomedical research tasks.
*   **LLM-Powered Reasoning:** Leverages the power of Large Language Models (LLMs) for intelligent reasoning and hypothesis generation.
*   **Retrieval-Augmented Planning:** Combines LLM capabilities with retrieval-augmented planning for enhanced task execution.
*   **Code-Based Execution:** Executes tasks through code, providing a flexible and powerful approach to biomedical research.
*   **Web Interface:** Access and experiment with Biomni's capabilities via a user-friendly, no-code web interface.
*   **MCP (Model Context Protocol) Support:** Integrate external tools via MCP servers.
*   **Open-Source & Community-Driven:** Biomni thrives on community contributions, offering opportunities for collaboration and innovation.

## Quick Start

### Installation

1.  **Set up the Environment:** Follow the instructions in the [biomni\_env/README.md](biomni_env/README.md) file to set up the software environment.
2.  **Activate the Environment:**
    ```bash
    conda activate biomni_e1
    ```
3.  **Install the Biomni Package:**
    ```bash
    pip install biomni --upgrade
    ```
    or for the latest updates
    ```bash
    pip install git+https://github.com/snap-stanford/Biomni.git@main
    ```
4.  **Configure API Keys:** Choose one of the following methods:

    <details>
    <summary>Click to expand - API Key Configuration</summary>

    #### Option 1: Using .env file (Recommended)

    Create a `.env` file in your project directory:

    ```bash
    # Copy the example file
    cp .env.example .env

    # Edit the .env file with your actual API keys
    ```

    Your `.env` file should look like:

    ```env
    # Required: Anthropic API Key for Claude models
    ANTHROPIC_API_KEY=your_anthropic_api_key_here

    # Optional: OpenAI API Key (if using OpenAI models)
    OPENAI_API_KEY=your_openai_api_key_here

    # Optional: Azure OpenAI API Key (if using Azure OpenAI models)
    OPENAI_API_KEY=your_azure_openai_api_key
    OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/

    # Optional: AI Studio Gemini API Key (if using Gemini models)
    GEMINI_API_KEY=your_gemini_api_key_here

    # Optional: groq API Key (if using groq as model provider)
    GROQ_API_KEY=your_groq_api_key_here

    # Optional: Set the source of your LLM for example:
    #"OpenAI", "AzureOpenAI", "Anthropic", "Ollama", "Gemini", "Bedrock", "Groq", "Custom"
    LLM_SOURCE=your_LLM_source_here

    # Optional: AWS Bedrock Configuration (if using AWS Bedrock models)
    AWS_BEARER_TOKEN_BEDROCK=your_bedrock_api_key_here
    AWS_REGION=us-east-1

    # Optional: Custom model serving configuration
    # CUSTOM_MODEL_BASE_URL=http://localhost:8000/v1
    # CUSTOM_MODEL_API_KEY=your_custom_api_key_here

    # Optional: Biomni data path (defaults to ./data)
    # BIOMNI_DATA_PATH=/path/to/your/data

    # Optional: Timeout settings (defaults to 600 seconds)
    # BIOMNI_TIMEOUT_SECONDS=600
    ```

    #### Option 2: Using shell environment variables

    Alternatively, configure your API keys in bash profile `~/.bashrc`:

    ```bash
    export ANTHROPIC_API_KEY="YOUR_API_KEY"
    export OPENAI_API_KEY="YOUR_API_KEY" # optional if you just use Claude
    export OPENAI_ENDPOINT="https://your-resource-name.openai.azure.com/" # optional unless you are using Azure
    export AWS_BEARER_TOKEN_BEDROCK="YOUR_BEDROCK_API_KEY" # optional for AWS Bedrock models
    export AWS_REGION="us-east-1" # optional, defaults to us-east-1 for Bedrock
    export GEMINI_API_KEY="YOUR_GEMINI_API_KEY" #optional if you want to use a gemini model
    export GROQ_API_KEY="YOUR_GROQ_API_KEY" # Optional: set this to use models served by Groq
    export LLM_SOURCE="Groq" # Optional: set this to use models served by Groq
    ```
    </details>

### Known Package Conflicts

Address potential package conflicts by reviewing the [docs/known\_conflicts.md](./docs/known_conflicts.md) file for a list of packages that may need manual installation.

### Basic Usage

```python
from biomni.agent import A1

# Initialize the agent with data path, Data lake will be automatically downloaded on first run (~11GB)
agent = A1(path='./data', llm='claude-sonnet-4-20250514')

# Execute biomedical tasks using natural language
agent.go("Plan a CRISPR screen to identify genes that regulate T cell exhaustion, generate 32 genes that maximize the perturbation effect.")
agent.go("Perform scRNA-seq annotation at [PATH] and generate meaningful hypothesis")
agent.go("Predict ADMET properties for this compound: CC(C)CC1=CC=C(C=C1)C(C)C(=O)O")
```
*If using Azure, prefix the model name with `azure-`*

## MCP (Model Context Protocol) Support

Integrate external tools using MCP servers:

```python
from biomni.agent import A1

agent = A1()
agent.add_mcp(config_path="./mcp_config.yaml")
agent.go("Find FDA active ingredient information for ibuprofen")
```

**Built-in MCP Servers:** Explore examples in [`tutorials/examples/add_mcp_server/`](tutorials/examples/add_mcp_server/) and [`tutorials/examples/expose_biomni_server/`](tutorials/examples/expose_biomni_server/) and learn more in the [MCP Integration Documentation](docs/mcp_integration.md).

## Contributing to Biomni

Join the open-science initiative! We welcome contributions in these areas:

*   New Tools
*   Datasets
*   Software Integration
*   Benchmarks
*   Tutorials, Examples, and Use Cases
*   Fixes/Improvements for Existing Tools

See the **[Contributing Guide](CONTRIBUTION.md)**.

You can also submit tool/database/software suggestions via [this form](https://forms.gle/nu2n1unzAYodTLVj6).

## Biomni-E2: Shaping the Future of Biomedical AI

Be a part of **Biomni-E2**! Contribute and be invited as a co-author on an upcoming paper, with all contributors acknowledged in our publications.

## Tutorials and Examples

*   **[Biomni 101](./tutorials/biomni_101.ipynb)** - Get started with basic concepts.
*   More tutorials are coming soon!

## 🌐 Web Interface

Explore Biomni's capabilities through our no-code web interface: **[biomni.stanford.edu](https://biomni.stanford.edu)**.

[![Watch the video](https://img.youtube.com/vi/E0BRvl23hLs/maxresdefault.jpg)](https://youtu.be/E0BRvl23hLs)

## Release schedule

*   \[ ] 8 Real-world research task benchmark/leaderboard release
*   \[ ] A tutorial on how to contribute to Biomni
*   \[ ] A tutorial on baseline agents
*   \[x] MCP support
*   \[x] Biomni A1+E1 release

## Important Notes

*   **Security:** Exercise caution as Biomni executes LLM-generated code with full system privileges. Use in isolated/sandboxed environments.
*   **Release Date:** This release was frozen as of April 15, 2025, and may differ from the web platform.
*   **Licensing:** Review licenses of integrated tools before commercial use, as they may have restrictive licenses.

## Cite Us

```
@article{huang2025biomni,
  title={Biomni: A General-Purpose Biomedical AI Agent},
  author={Huang, Kexin and Zhang, Serena and Wang, Hanchen and Qu, Yuanhao and Lu, Yingzhou and Roohani, Yusuf and Li, Ryan and Qiu, Lin and Zhang, Junze and Di, Yin and others},
  journal={bioRxiv},
  pages={2025--05},
  year={2025},
  publisher={Cold Spring Harbor Laboratory}
}