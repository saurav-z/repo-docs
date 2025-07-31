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

Biomni is an innovative, general-purpose biomedical AI agent designed to accelerate scientific discovery.  ([View the original repository](https://github.com/snap-stanford/Biomni))

## Key Features

*   **Autonomous Task Execution:** Biomni leverages large language models (LLMs) to autonomously execute complex research tasks.
*   **Retrieval-Augmented Planning:** Integrates retrieval-augmented planning for enhanced reasoning and decision-making.
*   **Code-Based Execution:** Utilizes code execution capabilities to generate and test hypotheses.
*   **Broad Application:** Designed for diverse biomedical subfields.
*   **Web Interface:** A user-friendly web interface is available for easy access and experimentation.

## Quick Start Guide

### Installation

1.  **Set up the Environment:** Follow the instructions in the [biomni_env/README.md](biomni_env/README.md) file to set up the necessary environment.
2.  **Activate the Environment:** Activate the environment using `conda activate biomni_e1`.
3.  **Install the Biomni Package:** Install the official Biomni package using `pip install biomni --upgrade`. For the latest updates, you can install from the GitHub source: `pip install git+https://github.com/snap-stanford/Biomni.git@main`.
4.  **Configure API Keys:** Configure your API keys using one of the following methods:

    *   **Option 1: Using .env file (Recommended)**

        *   Create a `.env` file in your project directory and populate it with your API keys:

        ```bash
        # Copy the example file
        cp .env.example .env

        # Edit the .env file with your actual API keys
        ```

        *   Your `.env` file should look like:

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

    *   **Option 2: Using shell environment variables**
        *   Configure your API keys in your bash profile `~/.bashrc`:

        ```bash
        export ANTHROPIC_API_KEY="YOUR_API_KEY"
        export OPENAI_API_KEY="YOUR_API_KEY" # optional if you just use Claude
        export AWS_BEARER_TOKEN_BEDROCK="YOUR_BEDROCK_API_KEY" # optional for AWS Bedrock models
        export AWS_REGION="us-east-1" # optional, defaults to us-east-1 for Bedrock
        export GEMINI_API_KEY="YOUR_GEMINI_API_KEY" #optional if you want to use a gemini model
        ```

#### ⚠️ Known Package Conflicts

*   Refer to [docs/known_conflicts.md](./docs/known_conflicts.md) for details on resolving package conflicts.

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

## Contributing to Biomni

Biomni is an open-source project that welcomes community contributions.

*   **New Tools:** Contribute specialized analysis functions and algorithms.
*   **Datasets:** Add curated biomedical data and knowledge bases.
*   **Software Integration:** Integrate existing biomedical software packages.
*   **Benchmarks:** Develop evaluation datasets and performance metrics.
*   **Tutorials and Examples:** Create helpful guides and use cases.
*   **Update existing tools:** Fix and improve current tools.

*   Refer to the **[Contributing Guide](CONTRIBUTION.md)** for detailed instructions.
*   Submit tool/database/software suggestions using [this form](https://forms.gle/nu2n1unzAYodTLVj6).

## Biomni-E2: Shaping the Future of Biomedical AI

Help us build **Biomni-E2**, the next-generation environment, and become a co-author on an upcoming publication!

*   **Significant Contributors:** Invited as co-authors on publications (e.g., 10+ significant tool contributions or equivalent).
*   **All Contributors:** Will be acknowledged in our publications.

## Tutorials and Examples

*   **[Biomni 101](./tutorials/biomni_101.ipynb)** - Basic concepts and first steps

## Web Interface

Explore Biomni through our no-code web interface at **[biomni.stanford.edu](https://biomni.stanford.edu)**.

[![Watch the video](https://img.youtube.com/vi/E0BRvl23hLs/maxresdefault.jpg)](https://youtu.be/E0BRvl23hLs)

## Release Schedule

*   [ ] 8 Real-world research task benchmark/leaderboard release
*   [ ] A tutorial on how to contribute to Biomni
*   [ ] A tutorial on baseline agents
*   [x] Biomni A1+E1 release

## Important Notes

*   **Security Warning:** Use in isolated/sandboxed environments due to potential risks from LLM-generated code.
*   **Frozen Release:**  This release was frozen as of April 15, 2025.
*   **Licensing:** Biomni is Apache 2.0-licensed, but integrated components may have other licenses.

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