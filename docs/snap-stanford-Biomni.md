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

**Biomni is a cutting-edge, general-purpose biomedical AI agent designed to accelerate scientific discovery.**  ([Original Repository](https://github.com/snap-stanford/Biomni))

## Key Features

*   **Autonomous Task Execution:** Biomni autonomously executes a wide range of research tasks across various biomedical subfields.
*   **LLM-Powered Reasoning:** Integrates advanced large language model (LLM) reasoning capabilities.
*   **Retrieval-Augmented Planning:** Utilizes retrieval-augmented planning for enhanced research efficiency.
*   **Code-Based Execution:** Leverages code-based execution to perform complex biomedical analyses.
*   **Enhances Research Productivity:** Helps scientists dramatically improve research output.
*   **Generates Testable Hypotheses:** Aids in the generation of new and testable scientific hypotheses.
*   **Web Interface:** Provides a user-friendly no-code web interface.
*   **Open Source and Community Driven:** Welcomes contributions to expand tools and datasets.

## Getting Started

### Installation

1.  **Set up the environment:** Follow the instructions in the [biomni_env/README.md](biomni_env/README.md) file to set up your environment.
2.  **Activate the environment:** `conda activate biomni_e1`
3.  **Install the Biomni package:**
    ```bash
    pip install biomni --upgrade
    ```
    or, for the latest updates:
    ```bash
    pip install git+https://github.com/snap-stanford/Biomni.git@main
    ```
4.  **Configure API Keys:** Add your API keys to your bash profile (`~/.bashrc`):
    ```bash
    export ANTHROPIC_API_KEY="YOUR_API_KEY"
    export OPENAI_API_KEY="YOUR_API_KEY" # optional if you just use Claude
    export AWS_BEARER_TOKEN_BEDROCK="YOUR_BEDROCK_API_KEY" # optional for AWS Bedrock models
    export AWS_REGION="us-east-1" # optional, defaults to us-east-1 for Bedrock
    ```

### Basic Usage

```python
from biomni.agent import A1

# Initialize the agent with data path
agent = A1(path='./data', llm='claude-sonnet-4-20250514')

# Execute tasks using natural language
agent.go("Plan a CRISPR screen to identify genes that regulate T cell exhaustion, generate 32 genes that maximize the perturbation effect.")
agent.go("Perform scRNA-seq annotation at [PATH] and generate meaningful hypothesis")
agent.go("Predict ADMET properties for this compound: CC(C)CC1=CC=C(C=C1)C(C)C(=O)O")
```

## Contributing

Biomni is an open-science initiative.  We welcome contributions in the following areas:

*   New Tools (analysis functions and algorithms)
*   Datasets (curated biomedical data and knowledge bases)
*   Software (integration of biomedical software packages)
*   Benchmarks (evaluation datasets and performance metrics)
*   Tutorials, Examples, and Use Cases
*   Updates to existing tools

Check out our **[Contributing Guide](CONTRIBUTION.md)** for more information.  You can also submit tool suggestions using [this form](https://forms.gle/nu2n1unzAYodTLVj6).

## Biomni-E2: The Future of Biomedical AI

Join us in building **Biomni-E2**, the next-generation environment developed *with and for the community*. Contributing to this project can lead to:

*   **Co-authorship:** Significant contributors may be invited to co-author a paper.
*   **Acknowledgement:** All contributors will be acknowledged in our publications.
*   **More contributor perks...**

## Tutorials and Examples

*   **[Biomni 101](./tutorials/biomni_101.ipynb)** - Basic concepts and first steps

## Web Interface

Explore Biomni's capabilities through our no-code web interface: **[biomni.stanford.edu](https://biomni.stanford.edu)**.

[![Watch the video](https://img.youtube.com/vi/E0BRvl23hLs/maxresdefault.jpg)](https://youtu.be/E0BRvl23hLs)

## Release Schedule (Roadmap)

*   [ ] 8 Real-world research task benchmark/leaderboard release
*   [ ] A tutorial on how to contribute to Biomni
*   [ ] A tutorial on baseline agents
*   [x] Biomni A1+E1 release

## Important Notes

*   **Security Warning:** Biomni executes LLM-generated code with full system privileges. Use it in isolated/sandboxed environments. The agent can access files, the network, and system commands. Exercise caution with sensitive data.
*   **Frozen Release:** This release was frozen as of April 15, 2025, and may differ from the current web platform.
*   **Licensing:** Biomni is Apache 2.0-licensed, but integrated tools/databases may have more restrictive licenses. Review each component carefully before commercial use.

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