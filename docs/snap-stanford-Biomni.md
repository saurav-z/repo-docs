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

**Biomni is a cutting-edge, general-purpose biomedical AI agent designed to accelerate scientific discovery across various biomedical fields.**

## Key Features

*   **Autonomous Task Execution:** Biomni intelligently executes complex research tasks with minimal human intervention.
*   **LLM-Powered Reasoning:** Leverages advanced large language models for sophisticated reasoning and planning.
*   **Retrieval-Augmented Planning:** Integrates retrieval-augmented generation (RAG) for enhanced accuracy and context-awareness.
*   **Code-Based Execution:** Executes tasks through code generation and execution, providing flexibility and reproducibility.
*   **Wide Range of Applications:** Suitable for diverse biomedical subfields, from genomics to drug discovery.
*   **Web Interface:** Try it out directly via the [Biomni Web UI](https://biomni.stanford.edu).

## Getting Started

### Installation

Follow the detailed instructions in the [biomni_env/README.md](biomni_env/README.md) file to set up the environment.

1.  **Activate the environment:**

    ```bash
    conda activate biomni_e1
    ```
2.  **Install the Biomni Python package:**

    ```bash
    pip install biomni --upgrade
    ```
    or, for the latest updates:

    ```bash
    pip install git+https://github.com/snap-stanford/Biomni.git@main
    ```
3.  **Configure API Keys:** Set your API keys in your `.bashrc` file:

    ```bash
    export ANTHROPIC_API_KEY="YOUR_API_KEY"
    export OPENAI_API_KEY="YOUR_API_KEY" # optional if you just use Claude
    ```

### Basic Usage

Once set up, you can begin using Biomni:

```python
from biomni.agent import A1

# Initialize the agent with a data path, data lake will be automatically downloaded on first run (~11GB)
agent = A1(path='./data', llm='claude-sonnet-4-20250514')

# Execute biomedical tasks using natural language
agent.go("Plan a CRISPR screen to identify genes that regulate T cell exhaustion, generate 32 genes that maximize the perturbation effect.")
agent.go("Perform scRNA-seq annotation at [PATH] and generate meaningful hypothesis")
agent.go("Predict ADMET properties for this compound: CC(C)CC1=CC=C(C=C1)C(C)C(=O)O")
```

## Contributing to Biomni

We welcome contributions to expand the capabilities of Biomni. Contribute to Biomni by:

*   **New Tools**: Develop new specialized analysis functions and algorithms.
*   **Datasets**: Curate and integrate biomedical data and knowledge bases.
*   **Software**: Integrate existing biomedical software packages.
*   **Benchmarks**: Create evaluation datasets and performance metrics.
*   **Tutorials & Examples**: Develop tutorials, examples, and use cases.
*   **Update existing tools**: Fix and enhance current tools.

Learn more about contributing in our [Contributing Guide](CONTRIBUTION.md).

You can also suggest new tool/database/software additions using [this form](https://forms.gle/nu2n1unzAYodTLVj6).

## Join the Biomni-E2 Initiative

Help us build **Biomni-E2**, the next-generation environment, designed **with and for the community**!

*   **Significant Contributors** will be invited as co-authors on our upcoming publications.
*   **All Contributors** will be acknowledged.

## Tutorials and Examples

**[Biomni 101](./tutorials/biomni_101.ipynb)** - A basic introduction to Biomni.

## Web Interface

Access Biomni through our user-friendly web interface: **[biomni.stanford.edu](https://biomni.stanford.edu)**.

[![Watch the video](https://img.youtube.com/vi/E0BRvl23hLs/maxresdefault.jpg)](https://youtu.be/E0BRvl23hLs)

## Release Schedule

*   [ ] 8 Real-world research task benchmark/leaderboard release
*   [ ] A tutorial on how to contribute to Biomni
*   [ ] A tutorial on baseline agents
*   [x] Biomni A1+E1 release

## Important Notes

*   This release was frozen as of April 15, 2025, and may differ from the current web platform.
*   Biomni is licensed under Apache 2.0. However, some integrated tools, databases, or software may have more restrictive licenses. Review each component carefully before any commercial use.

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

For more information, please visit the original repository: [https://github.com/snap-stanford/Biomni](https://github.com/snap-stanford/Biomni)