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

**Biomni is a groundbreaking, general-purpose biomedical AI agent designed to accelerate scientific discovery.** ([See the original repository](https://github.com/snap-stanford/Biomni))

## Key Features

*   **Autonomous Task Execution:** Biomni can autonomously perform a wide range of complex research tasks across various biomedical fields.
*   **LLM-Powered Reasoning:** Integrates cutting-edge Large Language Model (LLM) reasoning capabilities.
*   **Retrieval-Augmented Planning:** Leverages retrieval-augmented planning for more informed decision-making.
*   **Code-Based Execution:** Executes tasks through code, enabling complex analyses and simulations.
*   **Enhanced Research Productivity:** Dramatically improves research efficiency for scientists.
*   **Hypothesis Generation:** Facilitates the generation of testable hypotheses to drive discoveries.
*   **Web Interface:** A no-code web interface is available for easy access and exploration.

## Getting Started

### Installation

1.  **Setup Environment:** Follow the instructions in the [biomni\_env/README.md](biomni_env/README.md) file to set up the environment.
2.  **Activate Environment:** `conda activate biomni_e1`
3.  **Install Package:**  `pip install biomni --upgrade`
    *   For the latest updates, install from the GitHub source: `pip install git+https://github.com/snap-stanford/Biomni.git@main`
4.  **Configure API Keys:**  Set your API keys in your `.bashrc` file:

```bash
export ANTHROPIC_API_KEY="YOUR_API_KEY"
export OPENAI_API_KEY="YOUR_API_KEY" # optional if you just use Claude
export AWS_BEARER_TOKEN_BEDROCK="YOUR_BEDROCK_API_KEY" # optional for AWS Bedrock models
export AWS_REGION="us-east-1" # optional, defaults to us-east-1 for Bedrock
```

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

## Contribute to Biomni

We welcome contributions!  Help us build the future of biomedical AI.

*   **New Tools:** Develop specialized analysis functions and algorithms.
*   **Datasets:** Curate biomedical data and knowledge bases.
*   **Software Integration:** Integrate existing biomedical software packages.
*   **Benchmarks:** Create evaluation datasets and performance metrics.
*   **Tutorials & Examples:** Provide tutorials and usage examples.
*   **Improve Existing Tools:** Optimize and replace existing tools.

Check out the **[Contributing Guide](CONTRIBUTION.md)** for detailed instructions.

Alternatively, suggest tools, databases, or software via this [form](https://forms.gle/nu2n1unzAYodTLVj6).

## Join the Biomni-E2 Initiative

Be a part of building Biomni-E2, the next generation environment developed with and for the community!

*   **Co-authorship:** Contributors with significant impact (10+ tool contributions or equivalent) will be invited as co-authors.
*   **Acknowledgment:** All contributors will be acknowledged in our publications.
*   **More Contributor Perks:**  Additional benefits for contributors.

## Tutorials and Examples

*   **[Biomni 101](./tutorials/biomni_101.ipynb)** - Basic concepts and first steps

## Web Interface

Explore Biomni through our user-friendly web interface: **[biomni.stanford.edu](https://biomni.stanford.edu)**

[![Watch the video](https://img.youtube.com/vi/E0BRvl23hLs/maxresdefault.jpg)](https://youtu.be/E0BRvl23hLs)

## Release Schedule

*   [ ] 8 Real-world research task benchmark/leaderboard release
*   [ ] A tutorial on how to contribute to Biomni
*   [ ] A tutorial on baseline agents
*   [x] Biomni A1+E1 release

## Important Notes

*   **Security Warning:** Biomni currently executes LLM-generated code with full system privileges.  Use in isolated/sandboxed environments. The agent can access files, network, and system commands. Be careful with sensitive data or credentials.
*   **Release Freeze:** This release was frozen as of April 15, 2025, and may differ from the current web platform.
*   **Licensing:** Biomni is Apache 2.0-licensed; however, integrated tools/databases/software may have more restrictive licenses. Review each component carefully before commercial use.

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