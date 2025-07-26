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

**Biomni is a cutting-edge, general-purpose biomedical AI agent designed to empower researchers across various disciplines.**

## Key Features

*   **Autonomous Task Execution:** Biomni uses large language models (LLMs) with retrieval-augmented planning and code-based execution to autonomously handle a wide array of biomedical research tasks.
*   **Enhanced Research Productivity:** Drastically improve research efficiency and accelerate discovery by automating complex analyses and hypothesis generation.
*   **Versatile Applications:** Biomni supports diverse biomedical subfields, offering a flexible tool for various research needs.
*   **Web Interface:** Explore Biomni's capabilities through a user-friendly, no-code web interface [biomni.stanford.edu](https://biomni.stanford.edu).
*   **Open Source & Community Driven:** Biomni thrives on community contributions, fostering collaboration and innovation in biomedical AI.

## Getting Started

### Installation

1.  **Set up the Environment:** Follow the instructions in [biomni\_env/README.md](biomni_env/README.md) to configure your environment.
2.  **Activate the Environment:**

    ```bash
    conda activate biomni_e1
    ```
3.  **Install the Biomni Package:**

    ```bash
    pip install biomni --upgrade
    ```
    For the latest updates:
    ```bash
    pip install git+https://github.com/snap-stanford/Biomni.git@main
    ```
4.  **Configure API Keys:** Choose one of the following methods:

    *   **Option 1: Using .env file (Recommended)**

        *   Create a `.env` file in your project directory:

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

        Configure your API keys in your bash profile `~/.bashrc`:

        ```bash
        export ANTHROPIC_API_KEY="YOUR_API_KEY"
        export OPENAI_API_KEY="YOUR_API_KEY" # optional if you just use Claude
        export AWS_BEARER_TOKEN_BEDROCK="YOUR_BEDROCK_API_KEY" # optional for AWS Bedrock models
        export AWS_REGION="us-east-1" # optional, defaults to us-east-1 for Bedrock
        export GEMINI_API_KEY="YOUR_GEMINI_API_KEY" #optional if you want to use a gemini model
        ```
### Known Package Conflicts

Consult [docs/known_conflicts.md](./docs/known_conflicts.md) for a list of packages that may require manual installation due to dependency conflicts.

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

## Contribute

Biomni is an open-science project built by the community. Contribute to Biomni by:

*   **Developing New Tools**: Add analysis functions and algorithms
*   **Contributing Datasets**: Share curated biomedical data and knowledge bases.
*   **Integrating Existing Software**: Connect and utilize existing biomedical software packages.
*   **Creating Benchmarks**: Develop evaluation datasets and performance metrics.
*   **Providing Tutorials & Examples**: Build user-friendly tutorials and use case examples.
*   **Improving Existing Tools**: Fix and optimize existing features and functions.

See the **[Contributing Guide](CONTRIBUTION.md)** for details on how to contribute. Or, suggest a tool, database, or software to the Biomni team using [this form](https://forms.gle/nu2n1unzAYodTLVj6)

## Biomni-E2: The Future of Biomedical AI

We are building **Biomni-E2**, the next generation of Biomni, developed with and for the community. Be a part of it!

*   **Significant Contributors**: (10+ tool contributions or equivalent) will be co-authors on our upcoming publications.
*   **All Contributors**: Will be acknowledged in our publications.
*   **Join us** and help shape the future of biomedical AI agents.

## Tutorials and Examples

*   **[Biomni 101](./tutorials/biomni_101.ipynb)** - Learn the basics and get started.

## Web Interface

Try Biomni's user-friendly web interface at **[biomni.stanford.edu](https://biomni.stanford.edu)**.

[![Watch the video](https://img.youtube.com/vi/E0BRvl23hLs/maxresdefault.jpg)](https://youtu.be/E0BRvl23hLs)

## Release Status

*   \[ ] 8 Real-world research task benchmark/leaderboard release
*   \[ ] A tutorial on how to contribute to Biomni
*   \[ ] A tutorial on baseline agents
*   \[x] Biomni A1+E1 release

## Important Notes

*   **Security Warning:** Biomni executes LLM-generated code with full system privileges. Use it in isolated or sandboxed environments. Exercise caution with sensitive data.
*   **Release Version:** This release was frozen as of April 15, 2025, and may differ from the current web platform.
*   **Licensing:** Biomni is Apache 2.0-licensed. Review the licenses of integrated tools, databases, or software before commercial use.

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