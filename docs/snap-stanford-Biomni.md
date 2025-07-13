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

**Biomni is a general-purpose biomedical AI agent designed to accelerate scientific discovery across diverse biomedical fields.**  

**[View the Original Repository](https://github.com/snap-stanford/Biomni)**

## Key Features

*   **Autonomous Task Execution:** Automates complex biomedical research tasks using natural language instructions.
*   **LLM-Powered Reasoning:** Leverages cutting-edge Large Language Models (LLMs) for intelligent reasoning.
*   **Retrieval-Augmented Planning:** Enhances accuracy and context with retrieval-augmented planning.
*   **Code-Based Execution:** Executes tasks via code for reproducible and reliable results.
*   **Versatile Applications:** Designed for a wide range of tasks across various biomedical subfields.
*   **Web Interface:** Explore Biomni's capabilities through our user-friendly web interface at [biomni.stanford.edu](https://biomni.stanford.edu).

## Quick Start

### Installation

1.  **Set up the Environment:** Follow the instructions in the [biomni_env/README.md](biomni_env/README.md) file to set up the necessary software environment.
2.  **Activate the Environment:**
    ```bash
    conda activate biomni_e1
    ```
3.  **Install the Biomni Package:**
    ```bash
    pip install biomni --upgrade
    ```
    Alternatively, install from the GitHub source.
4.  **Configure API Keys:** Add your API keys to your `.bashrc` file:
    ```bash
    export ANTHROPIC_API_KEY="YOUR_API_KEY"
    export OPENAI_API_KEY="YOUR_API_KEY" # optional if you just use Claude
    ```

### Basic Usage

After setting up the environment, use Biomni in Python:

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

We welcome contributions from the community to enhance and expand Biomni. Contribute by:

*   **Adding New Tools:** Develop specialized analysis functions and algorithms.
*   **Curating Datasets:** Contribute biomedical data and knowledge bases.
*   **Integrating Software:** Integrate existing biomedical software packages.
*   **Developing Benchmarks:** Create evaluation datasets and performance metrics.
*   **Providing Documentation:** Write tutorials, examples, and use cases.
*   **Updating Existing Tools:** Improve and replace existing tools for enhanced performance.

See our **[Contributing Guide](CONTRIBUTION.md)** for detailed instructions.

Also, submit a tool/database/software suggestion via [this form](https://forms.gle/nu2n1unzAYodTLVj6).

## 🔬 Biomni-E2: Join the Next Generation

Help us build Biomni-E2, a next-generation environment developed with and for the community!

**Benefits for contributors:**

*   **Co-authorship:** Contributors with significant impact will be invited as co-authors on our upcoming paper.
*   **Recognition:** All contributors will be acknowledged in our publications.
*   **More perks**: stay tuned...

## Tutorials and Examples

*   **[Biomni 101](./tutorials/biomni_101.ipynb)** - Basic concepts and first steps
*   More tutorials coming soon!

## 🌐 Web Interface

Experience Biomni through our no-code web interface at **[biomni.stanford.edu](https://biomni.stanford.edu)**.

[![Watch the video](https://img.youtube.com/vi/E0BRvl23hLs/maxresdefault.jpg)](https://youtu.be/E0BRvl23hLs)

## Release Schedule

*   [ ] 8 Real-world research task benchmark/leaderboard release
*   [ ] A tutorial on how to contribute to Biomni
*   [ ] A tutorial on baseline agents
*   [x] Biomni A1+E1 release

## Important Notes

*   This release was frozen as of April 15, 2025, and may differ from the current web platform.
*   Biomni is licensed under Apache 2.0. Some integrated tools, databases, or software may have more restrictive licenses. Review the licenses of individual components before any commercial use.

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