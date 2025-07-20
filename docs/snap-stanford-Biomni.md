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

**Biomni is a cutting-edge, general-purpose biomedical AI agent designed to accelerate scientific discovery across diverse biomedical fields.**

## Key Features of Biomni

*   **Autonomous Task Execution:** Biomni intelligently executes a wide range of research tasks, from experimental planning to hypothesis generation.
*   **LLM-Powered Reasoning:** Integrates advanced large language models (LLMs) for sophisticated reasoning and problem-solving.
*   **Retrieval-Augmented Planning:** Leverages retrieval-augmented generation to enhance the planning process by accessing external knowledge and data.
*   **Code-Based Execution:** Utilizes code generation and execution capabilities to perform complex biomedical analyses.
*   **Web Interface:** Easily interact with Biomni through a user-friendly web interface for no-code experimentation.

## Getting Started

### Installation

1.  **Environment Setup:** Follow the instructions in the [biomni\_env/README.md](biomni_env/README.md) file to set up the required software environment.
2.  **Activate Environment:** Activate the environment using `conda activate biomni_e1`.
3.  **Install Biomni Package:** Install the Biomni Python package using one of the following methods:

    *   `pip install biomni --upgrade`
    *   `pip install git+https://github.com/snap-stanford/Biomni.git@main` (for the latest updates)
4.  **Configure API Keys:** Set your API keys in your `.bashrc` file:

    ```bash
    export ANTHROPIC_API_KEY="YOUR_API_KEY"
    export OPENAI_API_KEY="YOUR_API_KEY" # optional if you just use Claude
    export AWS_BEARER_TOKEN_BEDROCK="YOUR_BEDROCK_API_KEY" # optional for AWS Bedrock models
    export AWS_REGION="us-east-1" # optional, defaults to us-east-1 for Bedrock
    ```

### Basic Usage

Once the environment is set up, you can start using Biomni:

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

Biomni is an open-science initiative, and your contributions are valuable! We welcome contributions in the following areas:

*   🔧 New Tools: Develop specialized analysis functions and algorithms.
*   📊 Datasets: Contribute curated biomedical data and knowledge bases.
*   💻 Software: Integrate existing biomedical software packages.
*   📋 Benchmarks: Create evaluation datasets and performance metrics.
*   📚 Misc: Share tutorials, examples, and use cases.
*   🔧 Update Existing Tools: Fix and replace unoptimized tools.

Check out the **[Contributing Guide](CONTRIBUTION.md)** to learn how to contribute. You can also submit suggestions for new tools, databases, or software via [this form](https://forms.gle/nu2n1unzAYodTLVj6).

## Biomni-E2: Shaping the Future of Biomedical AI

Join us in building **Biomni-E2**, a next-generation environment designed **with and for the community** to accelerate scientific advancements.

*   **Significant Contributors:** Those with substantial contributions (e.g., 10+ integrated tools) will be co-authors on an upcoming paper.
*   **All Contributors:** Will be acknowledged in publications.

## Tutorials and Examples

*   **[Biomni 101](./tutorials/biomni_101.ipynb)** - Basic concepts and first steps

More tutorials are coming soon!

## 🌐 Web Interface

Explore Biomni's capabilities through our no-code web interface at **[biomni.stanford.edu](https://biomni.stanford.edu)**.

[![Watch the video](https://img.youtube.com/vi/E0BRvl23hLs/maxresdefault.jpg)](https://youtu.be/E0BRvl23hLs)

## Release Schedule

*   [ ] 8 Real-world research task benchmark/leaderboard release
*   [ ] A tutorial on how to contribute to Biomni
*   [ ] A tutorial on baseline agents
*   [x] Biomni A1+E1 release

## Important Notes

*   **Security Warning:** Biomni executes LLM-generated code with full system privileges.  Use it in isolated/sandboxed environments to prevent security risks. Be cautious with sensitive data.
*   **Release Freeze:** This release reflects the state as of April 15, 2025, and may differ from the current web platform.
*   **Licensing:** Biomni is Apache 2.0-licensed, but integrated components may have more restrictive licenses.  Review licenses carefully before commercial use.

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
```

[Back to the original repo](https://github.com/snap-stanford/Biomni)