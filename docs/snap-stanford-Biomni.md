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

# Biomni: Revolutionizing Biomedical Research with an AI Agent

**Biomni is a cutting-edge, general-purpose AI agent designed to empower scientists by automating complex biomedical research tasks.**

[View the original repository on GitHub](https://github.com/snap-stanford/Biomni)

## Key Features

*   **Autonomous Task Execution:** Biomni intelligently executes a wide array of biomedical research tasks across various subfields.
*   **LLM-Powered Reasoning:** Leverages the power of Large Language Models (LLMs) for advanced reasoning capabilities.
*   **Retrieval-Augmented Planning:** Enhances task execution with retrieval-augmented planning for improved accuracy and efficiency.
*   **Code-Based Execution:** Executes tasks through code, enabling complex analysis and hypothesis generation.
*   **Web Interface:** Easily accessible through a user-friendly web interface: [biomni.stanford.edu](https://biomni.stanford.edu)

## Getting Started

### Installation

1.  **Environment Setup:** Follow the instructions in the [biomni\_env/README.md](biomni_env/README.md) file to set up the required software environment.
2.  **Activate Environment:** Activate the environment using: `conda activate biomni_e1`.
3.  **Install Biomni Package:** Install the Biomni Python package using one of the following methods:
    *   `pip install biomni --upgrade` (for the latest release)
    *   `pip install git+https://github.com/snap-stanford/Biomni.git@main` (for the latest source code)
4.  **Configure API Keys:** Configure your API keys using either the `.env` file method (recommended) or shell environment variables:

    <details>
    <summary>API Key Configuration Details</summary>

    #### Option 1: Using .env file (Recommended)

    *   Create a `.env` file in your project directory and populate it with your API keys.

    ```bash
    # Copy the example file
    cp .env.example .env

    # Edit the .env file with your actual API keys
    ```

    *   Your `.env` file should include:

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

    *   Configure your API keys in your bash profile (`~/.bashrc`):

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

**Note:** If using Azure for your model, always prefix the model name with "azure-" (e.g., `llm='azure-gpt-4o'`).

## MCP (Model Context Protocol) Support

Biomni supports MCP servers for enhanced tool integration:

```python
from biomni.agent import A1

agent = A1()
agent.add_mcp(config_path="./mcp_config.yaml")
agent.go("Find FDA active ingredient information for ibuprofen")
```

**Built-in MCP Servers:** Refer to the [MCP Integration Documentation](docs/mcp_integration.md) and examples in [`tutorials/examples/add_mcp_server/`](tutorials/examples/add_mcp_server/) and [`tutorials/examples/expose_biomni_server/`](tutorials/examples/expose_biomni_server/) for details.

## Contribute to Biomni

Biomni is an open-science project, welcoming contributions from the community! We encourage contributions of:

*   🔧 New Tools
*   📊 Datasets
*   💻 Software Integrations
*   📋 Benchmarks
*   📚 Tutorials and Examples
*   🔧 Update existing tools

Refer to the **[Contributing Guide](CONTRIBUTION.md)** for detailed instructions.  You can also submit tool, database or software suggestions using [this form](https://forms.gle/nu2n1unzAYodTLVj6).

## Biomni-E2: Join the Next Generation

We are building **Biomni-E2** – a next-generation environment developed collaboratively with the community. Contributors with significant impact will be invited as co-authors on upcoming publications.  All contributors will be acknowledged.

## Tutorials and Examples

Explore the basic concepts and get started with Biomni with the following tutorial:

*   **[Biomni 101](./tutorials/biomni_101.ipynb)**

## Release Schedule

*   [ ] 8 Real-world research task benchmark/leaderboard release
*   [ ] A tutorial on how to contribute to Biomni
*   [ ] A tutorial on baseline agents
*   [x] MCP support
*   [x] Biomni A1+E1 release

## Important Notes

*   **Security Warning:** Currently, Biomni executes LLM-generated code with full system privileges.  Use in isolated/sandboxed environments to mitigate risks. Be cautious with sensitive data.
*   **Release Freeze:**  This release was frozen as of April 15, 2025.
*   **Licensing:** Biomni is Apache 2.0-licensed.  However, integrated tools and databases may have different licenses. Review each component's license before commercial use.

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