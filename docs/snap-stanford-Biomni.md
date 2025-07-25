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

Biomni is an innovative, general-purpose biomedical AI agent designed to empower researchers with automated task execution and hypothesis generation.  [Explore the original repository](https://github.com/snap-stanford/Biomni) for more details.

## Key Features

*   **Autonomous Task Execution:** Execute a wide range of biomedical research tasks across diverse subfields.
*   **LLM-Powered Reasoning:** Leverages advanced large language models (LLMs) for intelligent reasoning.
*   **Retrieval-Augmented Planning:** Integrates retrieval-augmented planning for enhanced task execution.
*   **Code-Based Execution:** Utilizes code execution for precise and reproducible results.
*   **Web Interface:** Experiment with Biomni through our user-friendly, no-code web interface.
*   **Open-Source & Community Driven:** Biomni thrives on community contributions.
*   **Hypothesis Generation:** Aids scientists in generating testable hypotheses to accelerate discoveries.

## Getting Started

### Installation

1.  **Set up the environment:** Follow the instructions in the [biomni_env/README.md](biomni_env/README.md) file.
2.  **Activate the environment:** `conda activate biomni_e1`
3.  **Install the Biomni package:**
    *   `pip install biomni --upgrade` or for the latest version: `pip install git+https://github.com/snap-stanford/Biomni.git@main`

### Configure API Keys

Configure your API keys using one of the following methods:

#### Option 1: .env file (Recommended)

1.  Create a `.env` file in your project directory:
    ```bash
    cp .env.example .env
    ```
2.  Edit the `.env` file with your API keys:

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

Configure API keys in your `.bashrc` file:

```bash
export ANTHROPIC_API_KEY="YOUR_API_KEY"
export OPENAI_API_KEY="YOUR_API_KEY" # optional if you just use Claude
export AWS_BEARER_TOKEN_BEDROCK="YOUR_BEDROCK_API_KEY" # optional for AWS Bedrock models
export AWS_REGION="us-east-1" # optional, defaults to us-east-1 for Bedrock
```

### Basic Usage

Once inside the environment:

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

Biomni welcomes contributions from the community! We're looking for contributions in:

*   New Tools
*   Datasets
*   Software integration
*   Benchmarks
*   Tutorials
*   Tool updates and fixes

See our **[Contributing Guide](CONTRIBUTION.md)** for details.

## Biomni-E2: Join the Next Generation

Help us build **Biomni-E2**, a next-generation environment.  Contributors with significant impact will be invited as co-authors, and all contributors will be acknowledged.

## Tutorials and Examples

*   **[Biomni 101](./tutorials/biomni_101.ipynb)** - Basic concepts and first steps

## Web Interface

Explore Biomni's capabilities through our web interface: **[biomni.stanford.edu](https://biomni.stanford.edu)**.

[![Watch the video](https://img.youtube.com/vi/E0BRvl23hLs/maxresdefault.jpg)](https://youtu.be/E0BRvl23hLs)

## Release Schedule

*   \[ ] 8 Real-world research task benchmark/leaderboard release
*   \[ ] A tutorial on how to contribute to Biomni
*   \[ ] A tutorial on baseline agents
*   [x] Biomni A1+E1 release

## Important Notes

*   **Security Warning:** Biomni executes LLM-generated code. Use in isolated/sandboxed environments.
*   This release was frozen as of April 15 2025.
*   Review licenses of integrated tools carefully.

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