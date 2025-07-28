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

# Biomni: Revolutionizing Biomedical Research with AI

Biomni is a powerful, general-purpose AI agent designed to accelerate biomedical discovery.  Explore the original repository at [https://github.com/snap-stanford/Biomni](https://github.com/snap-stanford/Biomni).

## Key Features

*   **Automated Research Tasks:** Execute a wide range of research tasks across diverse biomedical subfields.
*   **LLM-Powered Reasoning:** Leverages cutting-edge large language models (LLMs) for intelligent task execution.
*   **Retrieval-Augmented Planning:** Integrates retrieval-augmented planning for enhanced accuracy and context.
*   **Code-Based Execution:** Performs tasks through code execution, enabling complex analyses and simulations.
*   **Web Interface:**  Explore Biomni's capabilities through a user-friendly no-code web interface.

## Getting Started

### Installation

1.  **Set up the Environment:** Follow the instructions in the [biomni_env/README.md](biomni_env/README.md) file to set up the software environment.
2.  **Activate the Environment:**
    ```bash
    conda activate biomni_e1
    ```
3.  **Install the Biomni Package:**
    ```bash
    pip install biomni --upgrade
    ```
    or install from GitHub:
    ```bash
    pip install git+https://github.com/snap-stanford/Biomni.git@main
    ```
4.  **Configure API Keys:** Configure your API keys using one of the following methods:

    <details>
    <summary>Click to expand</summary>

    #### Option 1: Using `.env` File (Recommended)

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

    #### Option 2: Using Shell Environment Variables

    Configure your API keys in your bash profile (`~/.bashrc`):

    ```bash
    export ANTHROPIC_API_KEY="YOUR_API_KEY"
    export OPENAI_API_KEY="YOUR_API_KEY" # optional if you just use Claude
    export AWS_BEARER_TOKEN_BEDROCK="YOUR_BEDROCK_API_KEY" # optional for AWS Bedrock models
    export AWS_REGION="us-east-1" # optional, defaults to us-east-1 for Bedrock
    export GEMINI_API_KEY="YOUR_GEMINI_API_KEY" #optional if you want to use a gemini model
    ```
    </details>

### Known Package Conflicts
See the up-to-date list and details in [docs/known_conflicts.md](./docs/known_conflicts.md).

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

## Contributing to Biomni

Biomni thrives on community contributions! We welcome contributions in the following areas:

*   New Tools
*   Datasets
*   Software Integration
*   Benchmarks
*   Tutorials and Examples
*   Updates to Existing Tools

See the [Contributing Guide](CONTRIBUTION.md) for detailed instructions.  You can also submit suggestions via [this form](https://forms.gle/nu2n1unzAYodTLVj6).

## Biomni-E2: Join the Next Generation

We are building **Biomni-E2**, a community-driven environment.  Contribute and become a co-author!

## Tutorials and Examples

*   **[Biomni 101](./tutorials/biomni_101.ipynb)** - Basic concepts and first steps

More tutorials are coming soon!

## Web Interface

Explore Biomni through our user-friendly web interface:  **[biomni.stanford.edu](https://biomni.stanford.edu)**.

[![Watch the video](https://img.youtube.com/vi/E0BRvl23hLs/maxresdefault.jpg)](https://youtu.be/E0BRvl23hLs)

## Release Schedule
*   [ ] 8 Real-world research task benchmark/leaderboard release
*   [ ] A tutorial on how to contribute to Biomni
*   [ ] A tutorial on baseline agents
*   [x] Biomni A1+E1 release

## Important Notes
- **Security:** Biomni executes LLM-generated code with full system privileges. Use in isolated/sandboxed environments.  The agent can access files, network, and system commands.
- **Release Freeze:** This release was frozen as of April 15, 2025, and may differ from the current web platform.
- **Licensing:** Biomni is Apache 2.0-licensed.  Review the licenses of integrated tools, databases, and software before commercial use.

## Citation

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