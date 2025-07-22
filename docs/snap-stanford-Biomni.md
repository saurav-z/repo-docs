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

**Biomni is a cutting-edge, general-purpose AI agent designed to autonomously tackle complex biomedical research tasks, accelerating discovery and hypothesis generation.**  Learn more on the [original repository](https://github.com/snap-stanford/Biomni).

## Key Features

*   **Autonomous Task Execution:** Automates a wide range of research tasks across diverse biomedical subfields.
*   **LLM-Powered Reasoning:** Integrates advanced large language model (LLM) reasoning capabilities.
*   **Retrieval-Augmented Planning:** Leverages retrieval-augmented planning for enhanced accuracy and context.
*   **Code-Based Execution:** Executes tasks through code, enabling complex analyses and simulations.
*   **Web Interface:** Try it out with the no-code web interface: [biomni.stanford.edu](https://biomni.stanford.edu).

## Getting Started

### Installation

1.  **Set up the Environment:** Follow the instructions in the [biomni_env/README.md](biomni_env/README.md) file to set up your environment.
2.  **Activate the Environment:**
    ```bash
    conda activate biomni_e1
    ```
3.  **Install the Biomni Package:**
    ```bash
    pip install biomni --upgrade
    ```
    *   For the latest updates, install from the GitHub source:
        ```bash
        pip install git+https://github.com/snap-stanford/Biomni.git@main
        ```
4.  **Configure API Keys:** Choose one of the following methods:

    #### Option 1: .env File (Recommended)

    *   Create a `.env` file in your project directory:
        ```bash
        cp .env.example .env
        ```
    *   Edit `.env` with your API keys:
        ```env
        # Required: Anthropic API Key for Claude models
        ANTHROPIC_API_KEY=your_anthropic_api_key_here

        # Optional: OpenAI API Key (if using OpenAI models)
        OPENAI_API_KEY=your_openai_api_key_here

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

    #### Option 2: Shell Environment Variables

    *   Configure API keys in your bash profile (e.g., `~/.bashrc`):
        ```bash
        export ANTHROPIC_API_KEY="YOUR_API_KEY"
        export OPENAI_API_KEY="YOUR_API_KEY" # optional if you just use Claude
        export AWS_BEARER_TOKEN_BEDROCK="YOUR_BEDROCK_API_KEY" # optional for AWS Bedrock models
        export AWS_REGION="us-east-1" # optional, defaults to us-east-1 for Bedrock
        ```

### Basic Usage

Once inside the environment, you can start using Biomni:

```python
from biomni.agent import A1

# Initialize the agent with data path, Data lake will be automatically downloaded on first run (~11GB)
agent = A1(path='./data', llm='claude-sonnet-4-20250514')

# Execute biomedical tasks using natural language
agent.go("Plan a CRISPR screen to identify genes that regulate T cell exhaustion, generate 32 genes that maximize the perturbation effect.")
agent.go("Perform scRNA-seq annotation at [PATH] and generate meaningful hypothesis")
agent.go("Predict ADMET properties for this compound: CC(C)CC1=CC=C(C=C1)C(C)C(=O)O")
```

## Contributing

Biomni is an open-science project welcoming community contributions!  We encourage contributions in the following areas:

*   **New Tools:** Specialized analysis functions and algorithms
*   **Datasets:** Curated biomedical data and knowledge bases
*   **Software Integration:** Integration of existing biomedical software packages
*   **Benchmarks:** Evaluation datasets and performance metrics
*   **Tutorials and Examples:**  Guides and use cases
*   **Tool Improvements:** Fixes and optimizations to existing tools

  Check out this **[Contributing Guide](CONTRIBUTION.md)** on how to contribute to the Biomni ecosystem.

If you have particular tool/database/software in mind that you want to add, you can also submit to [this form](https://forms.gle/nu2n1unzAYodTLVj6) and the biomni team will implement them.

## Biomni-E2: The Next Generation

Join us in building **Biomni-E2**, the next-generation environment, developed *with* and *for* the community!

*   **Significant contributors** (e.g., 10+ impactful tool contributions) will be **invited as co-authors** on our upcoming publication.
*   **All contributors** will be acknowledged.

## Tutorials and Examples

*   **[Biomni 101](./tutorials/biomni_101.ipynb)** - Get started with the basics.

## Web Interface

Explore Biomni's capabilities through our user-friendly web interface: **[biomni.stanford.edu](https://biomni.stanford.edu)**.

[![Watch the video](https://img.youtube.com/vi/E0BRvl23hLs/maxresdefault.jpg)](https://youtu.be/E0BRvl23hLs)

## Release Schedule

*   [ ] 8 Real-world research task benchmark/leaderboard release
*   [ ] A tutorial on how to contribute to Biomni
*   [ ] A tutorial on baseline agents
*   [x] Biomni A1+E1 release

## Important Notes

*   **Security Warning:** Biomni executes LLM-generated code with full system privileges. Use it in isolated/sandboxed environments for production.
*   **Release Freeze:** This release was frozen as of April 15, 2025.
*   **Licensing:** Biomni is Apache 2.0-licensed. Review licenses for integrated tools/databases.

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