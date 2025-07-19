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

**Biomni is a cutting-edge, general-purpose AI agent designed to accelerate biomedical research by automating complex tasks.**

## Key Features:

*   **Automated Research Tasks:** Biomni can autonomously execute a wide range of biomedical research tasks across diverse subfields.
*   **LLM-Powered Reasoning:** Integrates advanced Large Language Models (LLMs) for sophisticated reasoning capabilities.
*   **Retrieval-Augmented Planning:** Utilizes retrieval-augmented planning to enhance task execution.
*   **Code-Based Execution:** Executes tasks through code, enabling complex analyses and simulations.
*   **Web Interface:**  Explore and utilize Biomni's capabilities through a user-friendly web interface.
*   **Open-Source & Community Driven**: Contribute to the development of Biomni and shape the future of biomedical AI.

## Getting Started

### Installation

1.  **Set up the Environment:** Follow the instructions in the [biomni\_env/README.md](biomni_env/README.md) file to set up the necessary environment.
2.  **Activate the Environment:**
    ```bash
    conda activate biomni_e1
    ```
3.  **Install the Biomni Package:**
    ```bash
    pip install biomni --upgrade
    ```
    Or, for the latest updates:
    ```bash
    pip install git+https://github.com/snap-stanford/Biomni.git@main
    ```
4.  **Configure API Keys:** Set your API keys in your `.bashrc` file:
    ```bash
    export ANTHROPIC_API_KEY="YOUR_API_KEY"
    export OPENAI_API_KEY="YOUR_API_KEY" # optional if you just use Claude
    export AWS_BEARER_TOKEN_BEDROCK="YOUR_BEDROCK_API_KEY" # optional for AWS Bedrock models
    export AWS_REGION="us-east-1" # optional, defaults to us-east-1 for Bedrock
    ```

### Basic Usage

Once the environment is set up, use Biomni in your Python scripts:

```python
from biomni.agent import A1

# Initialize the agent with data path
agent = A1(path='./data', llm='claude-sonnet-4-20250514')

# Execute biomedical tasks using natural language
agent.go("Plan a CRISPR screen to identify genes that regulate T cell exhaustion, generate 32 genes that maximize the perturbation effect.")
agent.go("Perform scRNA-seq annotation at [PATH] and generate meaningful hypothesis")
agent.go("Predict ADMET properties for this compound: CC(C)CC1=CC=C(C=C1)C(C)C(=O)O")
```

## Contribute to Biomni

Biomni thrives on community contributions. We welcome contributions in these areas:

*   **New Tools:** Develop specialized analysis functions and algorithms.
*   **Datasets:** Curate biomedical data and knowledge bases.
*   **Software Integration:** Integrate existing biomedical software packages.
*   **Benchmarks:** Provide evaluation datasets and performance metrics.
*   **Tutorials and Examples:** Create tutorials and examples.
*   **Tool Updates:** Improve and replace existing tools.

See our **[Contributing Guide](CONTRIBUTION.md)** for detailed instructions.

Have a specific tool, database, or software you want to add? Submit your idea via [this form](https://forms.gle/nu2n1unzAYodTLVj6).

## Biomni-E2: The Next Generation

Join us in building **Biomni-E2**, a collaborative environment designed with and for the community, to accelerate science. Contributors with significant impact will be invited as co-authors on an upcoming publication.

## Tutorials and Examples

*   **[Biomni 101](./tutorials/biomni_101.ipynb)** - Basic concepts and first steps
*   More tutorials are planned.

## Web Interface

Experience Biomni through our no-code web interface at **[biomni.stanford.edu](https://biomni.stanford.edu)**.

[![Watch the video](https://img.youtube.com/vi/E0BRvl23hLs/maxresdefault.jpg)](https://youtu.be/E0BRvl23hLs)

## Release Schedule

*   [ ] 8 Real-world research task benchmark/leaderboard release
*   [ ] A tutorial on how to contribute to Biomni
*   [ ] A tutorial on baseline agents
*   [x] Biomni A1+E1 release

## Important Notes

*   **Security:** Biomni executes LLM-generated code with full system privileges. Use in isolated/sandboxed environments.  Be cautious with sensitive data.
*   **Release Freeze:** This release reflects the state as of April 15, 2025.
*   **Licensing:** Biomni is Apache 2.0-licensed.  Review the licenses of integrated components.

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