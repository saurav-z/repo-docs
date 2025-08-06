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

# Biomni: Revolutionizing Biomedical Research with an AI Agent

**Biomni is a cutting-edge, general-purpose biomedical AI agent designed to accelerate scientific discovery across various subfields.**

## Key Features

*   **Autonomous Task Execution:** Automates complex research tasks, freeing up researchers' time.
*   **LLM-Powered Reasoning:** Leverages advanced large language models for intelligent decision-making.
*   **Retrieval-Augmented Planning:** Enhances planning capabilities with relevant data retrieval.
*   **Code-Based Execution:** Executes tasks by generating and running code.
*   **Wide Range of Applications:** Supports diverse biomedical research tasks.
*   **Web Interface:** Accessible and easy-to-use web interface for quick experimentation.
*   **Open Source and Community-Driven:** Active community with opportunities for contributions.

## Getting Started

### Installation

To get started with Biomni, follow these steps:

1.  **Set up the Environment:** Follow the instructions in the [biomni_env/README.md](biomni_env/README.md) file to set up the software environment.
2.  **Activate the Environment:**

    ```bash
    conda activate biomni_e1
    ```

3.  **Install the Biomni Package:**
    ```bash
    pip install biomni --upgrade
    ```
    Or install from the GitHub source:
    ```bash
    pip install git+https://github.com/snap-stanford/Biomni.git@main
    ```

4.  **Configure API Keys:**
    Choose one of the following methods to configure your API keys:

    *   **Option 1: Using `.env` File (Recommended)**
        *   Create a `.env` file in your project directory.
        *   Copy the example file: `cp .env.example .env`
        *   Edit the `.env` file with your API keys.  See the original README for details on setting API keys.
    *   **Option 2: Using Shell Environment Variables**
        *   Configure your API keys in your bash profile (`~/.bashrc` or similar). See the original README for details on setting API keys.

### Known Package Conflicts

Some Python packages are not installed by default due to dependency conflicts. Consult the [docs/known_conflicts.md](./docs/known_conflicts.md) for a current list and solutions.

### Basic Usage

Once the environment is set up and activated, you can start using Biomni:

```python
from biomni.agent import A1

# Initialize the agent with data path (Data lake will be automatically downloaded on first run (~11GB))
agent = A1(path='./data', llm='claude-sonnet-4-20250514')

# Execute biomedical tasks using natural language
agent.go("Plan a CRISPR screen to identify genes that regulate T cell exhaustion, generate 32 genes that maximize the perturbation effect.")
agent.go("Perform scRNA-seq annotation at [PATH] and generate meaningful hypothesis")
agent.go("Predict ADMET properties for this compound: CC(C)CC1=CC=C(C=C1)C(C)C(=O)O")
```
If you plan on using Azure for your model, always prefix the model name with azure- (e.g. llm='azure-gpt-4o').

## Model Context Protocol (MCP) Support

Biomni supports MCP servers for external tool integration, see the [MCP Integration Documentation](docs/mcp_integration.md) and examples in [`tutorials/examples/add_mcp_server/`](tutorials/examples/add_mcp_server/) and [`tutorials/examples/expose_biomni_server/`](tutorials/examples/expose_biomni_server/).

```python
from biomni.agent import A1

agent = A1()
agent.add_mcp(config_path="./mcp_config.yaml")
agent.go("Find FDA active ingredient information for ibuprofen")
```

## Contribute to Biomni

Biomni is an open-science initiative that welcomes community contributions. We encourage contributions in the following areas:

*   New Tools
*   Datasets
*   Software Integration
*   Benchmarks
*   Tutorials and Examples
*   Update existing tools

See the [Contributing Guide](CONTRIBUTION.md) for detailed instructions.

## Biomni-E2: Join the Next Generation

We are building **Biomni-E2**, a next-generation environment, with the community. Contribute and get invited as co-authors! All contributors will be acknowledged in our publications.

## Tutorials and Examples

Get started with our tutorials:

*   **[Biomni 101](./tutorials/biomni_101.ipynb)** - Basic concepts and first steps

## Web Interface

Try Biomni through our no-code web interface at **[biomni.stanford.edu](https://biomni.stanford.edu)**.

[![Watch the video](https://img.youtube.com/vi/E0BRvl23hLs/maxresdefault.jpg)](https://youtu.be/E0BRvl23hLs)

## Release Schedule

*   [ ] 8 Real-world research task benchmark/leaderboard release
*   [ ] A tutorial on how to contribute to Biomni
*   [ ] A tutorial on baseline agents
*   [x] MCP support
*   [x] Biomni A1+E1 release

## Important Notes

*   **Security:** Biomni executes LLM-generated code with full system privileges. Use in isolated/sandboxed environments for production.
*   **Version Freeze:** This release was frozen as of April 15, 2025, and may differ from the current web platform.
*   **Licensing:** Biomni is Apache 2.0-licensed; however, integrated tools may have more restrictive commercial licenses. Review each component before commercial use.

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

**[Back to Top](#)** - Back to the original repository.