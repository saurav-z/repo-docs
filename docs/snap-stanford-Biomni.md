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

**Biomni is a cutting-edge, general-purpose biomedical AI agent designed to accelerate scientific discovery by automating complex research tasks.**

[View the Biomni GitHub Repository](https://github.com/snap-stanford/Biomni)

## Key Features

*   **Automated Task Execution:** Execute a wide array of biomedical research tasks autonomously.
*   **LLM-Powered Reasoning:** Leverages Large Language Models (LLMs) for advanced reasoning capabilities.
*   **Retrieval-Augmented Planning:** Integrates retrieval-augmented planning for enhanced task execution.
*   **Code-Based Execution:** Utilizes code-based execution for accurate and reproducible results.
*   **Web Interface:** Accessible via a user-friendly web interface for easy experimentation and collaboration.
*   **Open Source and Community Driven**: Biomni is open-source and welcomes community contributions, fostering innovation and collaboration.

## Quick Start Guide

### Installation

Setting up the Biomni environment involves a few key steps.  First, ensure you have set up the environment using the provided `setup.sh` script located in the [biomni\_env/README.md](biomni_env/README.md) file.

1.  **Activate Environment:**
    ```bash
    conda activate biomni_e1
    ```
2.  **Install Biomni Package:**
    ```bash
    pip install biomni --upgrade
    ```
    Alternatively, for the latest updates from the GitHub source:
    ```bash
    pip install git+https://github.com/snap-stanford/Biomni.git@main
    ```
3.  **Configure API Keys:**  Set your API keys in your bash profile (`~/.bashrc`):
    ```bash
    export ANTHROPIC_API_KEY="YOUR_API_KEY"
    export OPENAI_API_KEY="YOUR_API_KEY" # optional if you just use Claude
    ```

### Basic Usage

Once your environment is set up, you can start using Biomni with the following example:

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

Biomni is an open-science initiative that welcomes community contributions of all kinds! Help us improve Biomni by contributing:

*   🔧 **New Tools:** Specialized analysis functions and algorithms
*   📊 **Datasets:** Curated biomedical data and knowledge bases
*   💻 **Software:** Integration of existing biomedical software packages
*   📋 **Benchmarks:** Evaluation datasets and performance metrics
*   📚 **Tutorials and Examples:** Help others learn how to use Biomni
*   🔧 **Improve Existing Tools:** Fix and optimize existing tools.

Check out our **[Contributing Guide](CONTRIBUTION.md)** for detailed instructions.

Have a specific tool, database, or software in mind?  Submit your ideas using [this form](https://forms.gle/nu2n1unzAYodTLVj6).

## Join the Biomni-E2 Initiative

Be a part of Biomni-E2, the next-generation environment developed with and for the community!

*   **Significant Contributors** (10+ impactful tool contributions) will be invited as co-authors on upcoming publications.
*   **All Contributors** will be acknowledged in publications.
*   More contributor perks are available - join us!

## Tutorials and Examples

*   **[Biomni 101](./tutorials/biomni_101.ipynb)** - Basic concepts and first steps
*   More tutorials are planned, so stay tuned!

## 🌐 Web Interface

Try Biomni now through our no-code web interface: **[biomni.stanford.edu](https://biomni.stanford.edu)**.

[![Watch the video](https://img.youtube.com/vi/E0BRvl23hLs/maxresdefault.jpg)](https://youtu.be/E0BRvl23hLs)

## Release Schedule and Future Updates

*   [ ] 8 Real-world research task benchmark/leaderboard release
*   [ ] A tutorial on how to contribute to Biomni
*   [ ] A tutorial on baseline agents
*   [x] Biomni A1+E1 release

## Important Notes

*   This release reflects the version as of April 15, 2025, and may differ from the current web platform.
*   Biomni is licensed under Apache 2.0, but integrated tools and databases may have more restrictive commercial licenses. Always review the licenses of individual components before commercial use.

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