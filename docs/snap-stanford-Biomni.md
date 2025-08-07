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

**Biomni is a groundbreaking general-purpose AI agent designed to automate complex research tasks and accelerate discoveries across diverse biomedical fields.**

## Key Features

*   **Autonomous Task Execution:** Biomni uses large language models (LLMs) to understand and execute a wide array of biomedical research tasks.
*   **Retrieval-Augmented Planning:** Enhances LLM capabilities by integrating retrieval-augmented planning for more accurate and relevant results.
*   **Code-Based Execution:** Executes code to perform complex analyses, simulations, and data manipulations.
*   **Versatile Applications:** Suitable for a broad range of biomedical subfields, from genomics to drug discovery.
*   **Web Interface:** Accessible and easy to use through a web interface [biomni.stanford.edu](https://biomni.stanford.edu).

## Getting Started

### Installation

1.  **Environment Setup:** Follow the instructions in the [biomni\_env/README.md](biomni_env/README.md) file to set up the necessary environment.
2.  **Activate Environment:** Activate the environment:

    ```bash
    conda activate biomni_e1
    ```

3.  **Install Biomni Package:**

    ```bash
    pip install biomni --upgrade
    ```

    Or, for the latest updates from the GitHub repository:

    ```bash
    pip install git+https://github.com/snap-stanford/Biomni.git@main
    ```

4.  **Configure API Keys:** Configure API keys using one of the following methods:

    *   **Option 1: .env file (Recommended)**
        *   Create a `.env` file in your project directory:
            ```bash
            # Copy the example file
            cp .env.example .env
            ```
        *   Edit the `.env` file with your API keys. See the original README for details.
    *   **Option 2: Shell environment variables**
        *   Set the API keys in your bash profile (`~/.bashrc`). See the original README for details.

    **Note:** See the [docs/known\_conflicts.md](./docs/known_conflicts.md) file for known package conflicts and workarounds.

### Basic Usage

Once your environment is set up:

```python
from biomni.agent import A1

# Initialize the agent with data path, Data lake will be automatically downloaded on first run (~11GB)
agent = A1(path='./data', llm='claude-sonnet-4-20250514')

# Execute biomedical tasks using natural language
agent.go("Plan a CRISPR screen to identify genes that regulate T cell exhaustion, generate 32 genes that maximize the perturbation effect.")
agent.go("Perform scRNA-seq annotation at [PATH] and generate meaningful hypothesis")
agent.go("Predict ADMET properties for this compound: CC(C)CC1=CC=C(C=C1)C(C)C(=O)O")
```

**Important:** If using Azure for your model, always prefix the model name with `azure-` (e.g., `llm='azure-gpt-4o'`).

## Model Context Protocol (MCP) Support

Biomni supports MCP servers for seamless integration with external tools:

```python
from biomni.agent import A1

agent = A1()
agent.add_mcp(config_path="./mcp_config.yaml")
agent.go("Find FDA active ingredient information for ibuprofen")
```

**Built-in MCP Servers:**

For usage and implementation details, refer to the [MCP Integration Documentation](docs/mcp_integration.md) and examples in [`tutorials/examples/add_mcp_server/`](tutorials/examples/add_mcp_server/) and [`tutorials/examples/expose_biomni_server/`](tutorials/examples/expose_biomni_server/).

## Contribute to Biomni

**Join the open-science initiative and help build the future of biomedical AI!** We welcome contributions in the following areas:

*   🔧 **New Tools:** Develop specialized analysis functions and algorithms.
*   📊 **Datasets:** Contribute curated biomedical data and knowledge bases.
*   💻 **Software:** Integrate existing biomedical software packages.
*   📋 **Benchmarks:** Create evaluation datasets and performance metrics.
*   📚 **Misc:**  Develop tutorials, examples, and use cases.
*   🔧 **Update existing tools**: Refactor and improve existing tools!

See the **[Contributing Guide](CONTRIBUTION.md)** for details on how to contribute.

You can also submit suggestions for tools/databases/software to [this form](https://forms.gle/nu2n1unzAYodTLVj6).

## Biomni-E2: The Next Generation

We're building **Biomni-E2** – a next-generation environment developed **with and for the community**.
**Join us in shaping the future of biomedical AI agent.**

*   Contributors with significant impact will be invited as co-authors on our upcoming publications.
*   All contributors will be acknowledged in our publications.

## Tutorials and Examples

*   **[Biomni 101](./tutorials/biomni_101.ipynb)** - Basic concepts and first steps

More to come!

## Web Interface

Experience Biomni through our no-code web interface at **[biomni.stanford.edu](https://biomni.stanford.edu)**.

[![Watch the video](https://img.youtube.com/vi/E0BRvl23hLs/maxresdefault.jpg)](https://youtu.be/E0BRvl23hLs)

## Release Schedule

*   [ ] 8 Real-world research task benchmark/leaderboard release
*   [ ] A tutorial on how to contribute to Biomni
*   [ ] A tutorial on baseline agents
*   [x] MCP support
*   [x] Biomni A1+E1 release

## Important Notes and Security

*   **Security Warning:** Biomni executes LLM-generated code with full system privileges. Use it in isolated/sandboxed environments. The agent can access files, network, and system commands. Be careful with sensitive data or credentials.
*   This release was frozen as of April 15 2025.
*   Biomni is Apache 2.0-licensed, but certain integrated tools may carry more restrictive commercial licenses. Review each component carefully.

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

[Back to Top](#biomni-revolutionizing-biomedical-research-with-ai)