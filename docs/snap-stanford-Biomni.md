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

**Biomni is a groundbreaking general-purpose biomedical AI agent designed to accelerate scientific discovery.**

## Key Features

*   **Autonomous Research Tasks:** Execute a wide array of biomedical tasks across diverse subfields.
*   **LLM-Powered Reasoning:** Leverages cutting-edge large language models for intelligent decision-making.
*   **Retrieval-Augmented Planning:** Enhances research accuracy and efficiency through data retrieval and planning.
*   **Code-Based Execution:** Enables precise and reproducible results through code execution.
*   **Web Interface:**  Explore Biomni's capabilities with a user-friendly, no-code web interface.

## Quick Start

### Installation

1.  **Environment Setup:** Follow the instructions in the [biomni\_env/README.md](biomni_env/README.md) file to set up your environment. This includes a large, pre-configured software environment.

2.  **Activate Environment:**

    ```bash
    conda activate biomni_e1
    ```

3.  **Install Biomni Package:**

    ```bash
    pip install biomni --upgrade
    ```
    Or, install from the GitHub source.

4.  **Configure API Keys:** Add your API keys to your `.bashrc` file:

    ```bash
    export ANTHROPIC_API_KEY="YOUR_API_KEY"
    export OPENAI_API_KEY="YOUR_API_KEY" # optional if you just use Claude
    ```

### Basic Usage

After activating the environment, use Biomni with Python:

```python
from biomni.agent import A1

# Initialize the agent with data path, Data lake will be automatically downloaded on first run (~11GB)
agent = A1(path='./data', llm='claude-sonnet-4-20250514')

# Execute biomedical tasks using natural language
agent.go("Plan a CRISPR screen to identify genes that regulate T cell exhaustion, generate 32 genes that maximize the perturbation effect.")
agent.go("Perform scRNA-seq annotation at [PATH] and generate meaningful hypothesis")
agent.go("Predict ADMET properties for this compound: CC(C)CC1=CC=C(C=C1)C(C)C(=O)O")
```

## Contribute to Biomni

Biomni is an open-science initiative and welcomes community contributions. Join us in building the future of biomedical AI!

*   **New Tools:**  Implement specialized analysis functions and algorithms.
*   **Datasets:**  Contribute curated biomedical data and knowledge bases.
*   **Software Integration:**  Integrate existing biomedical software packages.
*   **Benchmarks:**  Develop evaluation datasets and performance metrics.
*   **Tutorials and Examples:**  Create tutorials, examples, and use cases.
*   **Update Existing Tools:** Improve existing tools or suggest replacements.

See our **[Contributing Guide](CONTRIBUTION.md)** to learn how to contribute, or submit tool/database/software suggestions via [this form](https://forms.gle/nu2n1unzAYodTLVj6).

## Biomni-E2: Join the Next Generation

Help us build **Biomni-E2**, a community-driven environment designed to accelerate biomedical research.  Contributors will be:

*   **Invited as co-authors** (for significant contributions) on upcoming publications.
*   **Acknowledged** in our publications.
*   Offered other contributor perks.

## Tutorials and Examples

*   **[Biomni 101](./tutorials/biomni_101.ipynb)** - Get started with the basics.

## Web Interface

Experience Biomni's power through our web interface: **[biomni.stanford.edu](https://biomni.stanford.edu)**.

[![Watch the video](https://img.youtube.com/vi/E0BRvl23hLs/maxresdefault.jpg)](https://youtu.be/E0BRvl23hLs)

## Release Schedule

*   \[ ] 8 Real-world research task benchmark/leaderboard release
*   \[ ] A tutorial on how to contribute to Biomni
*   \[ ] A tutorial on baseline agents
*   \[x] Biomni A1+E1 release

## Important Notes

*   This release reflects the state of Biomni as of April 15, 2025.
*   Biomni is licensed under Apache 2.0.  Review individual components for their specific licensing before commercial use.

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

**[Explore the Biomni Repository on GitHub](https://github.com/snap-stanford/Biomni)**