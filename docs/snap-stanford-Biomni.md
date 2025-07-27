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

**Biomni is a cutting-edge, general-purpose biomedical AI agent designed to autonomously tackle complex research tasks, accelerating scientific discovery.** Find the original repository [here](https://github.com/snap-stanford/Biomni).

## Key Features

*   **Autonomous Task Execution:** Biomni autonomously executes a wide array of research tasks across diverse biomedical subfields.
*   **LLM-Powered Reasoning:** Integrates large language model (LLM) reasoning for intelligent task planning and execution.
*   **Retrieval-Augmented Planning:** Employs retrieval-augmented planning to access and leverage relevant biomedical knowledge.
*   **Code-Based Execution:** Executes code for precise and reproducible scientific analysis.
*   **Web Interface:** Explore Biomni's capabilities through a user-friendly, no-code web interface.

## Quickstart

### Installation

1.  **Set up the Environment:** Follow the instructions in the [biomni\_env/README.md](biomni_env/README.md) file to configure your environment.
2.  **Activate the Environment:**

    ```bash
    conda activate biomni_e1
    ```
3.  **Install the Biomni Package:**

    ```bash
    pip install biomni --upgrade
    ```
    or install from the github source version:

    ```bash
    pip install git+https://github.com/snap-stanford/Biomni.git@main
    ```
4.  **Configure API Keys:** Choose one of the following methods to configure your API keys:

    <details>
    <summary>Expand for API Key Configuration</summary>

    #### Option 1: Using .env File (Recommended)

    1.  Create a `.env` file in your project directory:
        ```bash
        # Copy the example file
        cp .env.example .env
        ```
    2.  Edit the `.env` file to include your API keys:
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

*   **Important:** Some Python packages are not installed by default in the Biomni environment due to dependency conflicts. See [docs/known\_conflicts.md](./docs/known_conflicts.md) for the up-to-date list and details.

### Basic Usage

Once inside the environment, use Biomni by importing `A1`:

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

Biomni is an open-science initiative.  We welcome contributions, including:

*   New Tools and Analysis Functions
*   Curated Datasets and Knowledge Bases
*   Integration of Biomedical Software
*   Evaluation Datasets and Metrics
*   Tutorials, Examples, and Use Cases
*   Updates to Existing Tools

See the **[Contributing Guide](CONTRIBUTION.md)** for details.

To suggest a tool/database/software, submit to [this form](https://forms.gle/nu2n1unzAYodTLVj6).

## Biomni-E2: Building the Future Together

Join us in shaping **Biomni-E2**, a next-generation environment developed *with and for the community* to accelerate science.

**Contributors with significant impact** can be **invited as co-authors** on upcoming publications, and all contributors will be acknowledged.

Let's build it together.

## Tutorials and Examples

*   **[Biomni 101](./tutorials/biomni_101.ipynb)** - Basic concepts and first steps

More tutorials are coming soon!

## 🌐 Web Interface

Experience Biomni through our no-code web interface at **[biomni.stanford.edu](https://biomni.stanford.edu)**.

[![Watch the video](https://img.youtube.com/vi/E0BRvl23hLs/maxresdefault.jpg)](https://youtu.be/E0BRvl23hLs)

## Release Schedule

*   \[ ] 8 Real-world research task benchmark/leaderboard release
*   \[ ] A tutorial on how to contribute to Biomni
*   \[ ] A tutorial on baseline agents
*   \[x] Biomni A1+E1 release

## Important Notes

*   **Security Warning:** Biomni executes LLM-generated code with full system privileges. Use it in isolated/sandboxed environments to protect against potential security risks.
*   **Release Freeze:** The current release was frozen as of April 15, 2025, and may differ from the live web platform.
*   **Licensing:** Biomni is Apache 2.0-licensed, but integrated components may have more restrictive licenses. Review each component before commercial use.

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