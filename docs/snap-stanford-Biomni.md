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

# Biomni: Your AI Assistant for Biomedical Research

**Biomni is a cutting-edge, general-purpose AI agent designed to revolutionize biomedical research by autonomously executing complex tasks and generating novel hypotheses.** [Explore the original repository](https://github.com/snap-stanford/Biomni).

## Key Features:

*   **Autonomous Task Execution:** Automates a wide range of research tasks across various biomedical subfields.
*   **LLM-Powered Reasoning:** Leverages state-of-the-art Large Language Models (LLMs) for intelligent reasoning and planning.
*   **Retrieval-Augmented Planning:** Enhances accuracy and efficiency through integrated retrieval mechanisms.
*   **Code-Based Execution:** Executes tasks through generated code, enabling complex analysis and simulations.
*   **Web Interface:**  Experience Biomni through our user-friendly no-code web interface.
*   **Open Source & Community Driven:** Biomni is built with open science principles and welcomes community contributions.
*   **MCP Support:** Supports Model Context Protocol (MCP) servers for seamless external tool integration.

## Getting Started

### Installation

1.  **Environment Setup:** Follow the instructions in the [biomni\_env/README.md](biomni_env/README.md) file to set up the necessary environment.
2.  **Activate Environment:** Activate the environment using: `conda activate biomni_e1`
3.  **Install Package:** Install the Biomni package using pip:
    *   Recommended: `pip install biomni --upgrade`
    *   Alternative: `pip install git+https://github.com/snap-stanford/Biomni.git@main`
4.  **Configure API Keys:** Configure your API keys using one of the following methods:

    <details>
    <summary>Click to expand</summary>

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

    # Optional: AI Studio Gemini API Key (if using Gemini models)
    GEMINI_API_KEY=your_gemini_api_key_here

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
    export AWS_BEARER_TOKEN_BEDROCK="YOUR_BEDROCK_API_KEY" # optional for AWS Bedrock models
    export AWS_REGION="us-east-1" # optional, defaults to us-east-1 for Bedrock
    export GEMINI_API_KEY="YOUR_GEMINI_API_KEY" #optional if you want to use a gemini model
    ```
    </details>

    #### ⚠️ Known Package Conflicts

    Some Python packages are not installed by default in the Biomni environment due to dependency conflicts. If you need these features, you must install the packages manually and may need to uncomment relevant code in the codebase. See the up-to-date list and details in [docs/known_conflicts.md](./docs/known_conflicts.md).

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

## MCP (Model Context Protocol) Support

Integrate external tools with Biomni using MCP servers.

```python
from biomni.agent import A1

agent = A1()
agent.add_mcp(config_path="./mcp_config.yaml")
agent.go("Find FDA active ingredient information for ibuprofen")
```

**Built-in MCP Servers:**  Refer to the [MCP Integration Documentation](docs/mcp_integration.md) and examples in [`tutorials/examples/add_mcp_server/`](tutorials/examples/add_mcp_server/) and [`tutorials/examples/expose_biomni_server/`](tutorials/examples/expose_biomni_server/) for details.

## Contribute to Biomni

Join our open-science initiative and help shape the future of biomedical AI! We welcome contributions in the following areas:

*   🔧 New Tools
*   📊 Datasets
*   💻 Software Integration
*   📋 Benchmarks
*   📚 Tutorials and Examples
*   🔧 Update existing tools

See our **[Contributing Guide](CONTRIBUTION.md)** for details.

Want to add a specific tool or database? Submit your suggestions using [this form](https://forms.gle/nu2n1unzAYodTLVj6).

## Biomni-E2: The Next Generation

Be a part of building **Biomni-E2** - the next-generation environment developed *with and for the community*.

**Contributors with significant impact** (e.g., 10+ significant & integrated tool contributions or equivalent) will be **invited as co-authors** on our upcoming paper in a top-tier journal or conference.

**All contributors** will be acknowledged in our publications.

Let’s build it together.

## Tutorials and Examples

*   **[Biomni 101](./tutorials/biomni_101.ipynb)** - Basic concepts and first steps

More to come!

## Web Interface

Try Biomni through our user-friendly web interface at **[biomni.stanford.edu](https://biomni.stanford.edu)**.

[![Watch the video](https://img.youtube.com/vi/E0BRvl23hLs/maxresdefault.jpg)](https://youtu.be/E0BRvl23hLs)

## Release Schedule

*   [ ] 8 Real-world research task benchmark/leaderboard release
*   [ ] A tutorial on how to contribute to Biomni
*   [ ] A tutorial on baseline agents
*   [x] MCP support
*   [x] Biomni A1+E1 release

## Important Notes

*   **Security Warning:** Biomni executes LLM-generated code with full system privileges. Use in isolated/sandboxed environments for production.  Be cautious with sensitive data and credentials.
*   **Release Freeze:** This release was frozen as of April 15, 2025, and may differ from the current web platform.
*   **Licensing:** Biomni is Apache 2.0-licensed.  Review the licenses of integrated tools, databases, and software before commercial use.

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
```