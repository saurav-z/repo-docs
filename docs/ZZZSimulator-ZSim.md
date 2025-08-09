# ZSim: Zenless Zone Zero Battle Simulator & Damage Calculator

**Unleash the full potential of your team in Zenless Zone Zero with ZSim, the ultimate battle simulator and damage calculator.**

[Link to Original Repo: ZSim on GitHub](https://github.com/ZZZSimulator/ZSim)

![ZSim Project Logo](docs/img/横板logo成图.png)

## Introduction

ZSim is a powerful tool designed for **Zenless Zone Zero** (ZZZ) players, offering a comprehensive battle simulation and damage calculation experience.  It allows you to analyze team compositions and optimize your gameplay strategy.  With ZSim, you can easily evaluate damage output, understand character interactions, and fine-tune your builds without the need to manually input skill sequences (unless you want them!).  Simply configure your agents' equipment, choose an Action Priority List (APL), and let ZSim automatically simulate battles, providing detailed results and insightful visualizations.

## Key Features

*   **Automated Battle Simulation:**  Simulates battles based on pre-set APLs, eliminating manual sequence input.
*   **Comprehensive Damage Calculation:** Calculates total damage output for a team, considering character equipment.
*   **Visual Reporting:** Generates intuitive charts and tables for easy analysis of results.
*   **Detailed Character Analysis:** Provides in-depth damage information for each character in your team.
*   **Equipment Customization:**  Allows you to easily edit and configure your agents' equipment.
*   **APL Editing:**  Offers the ability to customize Action Priority Lists for strategic control.

## Installation

To get started with ZSim, follow these steps:

1.  **Download:** Download the latest source code from the [release page](link to releases on github, if available) or clone the repository using `git clone`.
2.  **Install UV (if you haven't already):** UV is a fast package manager for Python. Open your terminal and run one of the following commands based on your operating system:

    ```bash
    # Using pip if you have python installed:
    pip install uv
    ```

    ```bash
    # On macOS or Linux:
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

    ```bash
    # On Windows11 24H2 or later:
    winget install --id=astral-sh.uv  -e
    ```

    ```bash
    # On lower version of Windows:
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    ```
    Or check the official installation guide: [https://docs.astral.sh/uv/getting-started/installation/](https://docs.astral.sh/uv/getting-started/installation/)
3.  **Run ZSim:** Open your terminal in the project directory and execute:

    ```bash
    uv sync
    uv run zsim run
    ```

## Development

### Key Components

*   **Simulation Engine:**  `zsim/simulator/` (Core battle simulation logic)
*   **Web API:**  `zsim/api_src/` (FastAPI-based REST API)
*   **Web UI:**  `zsim/webui.py` (Streamlit-based interface) and `electron-app/` (Vue.js + Electron desktop application)
*   **CLI:**  `zsim/run.py` (Command-line interface)
*   **Database:** SQLite-based storage for character/enemy data
*   **Electron App:** Desktop app built with Vue.js and Electron (communicates with the FastAPI backend)

### Setup and Installation

```bash
# Install UV package manager first
uv sync
# For WebUI development
uv run zsim run 
# For FastAPI backend
uv run zsim api

# For Electron App development, also install Node.js dependencies
cd electron-app
corepack install
pnpm install
```

### Testing Structure

*   Unit tests in `tests/`
*   API tests in `tests/api/`
*   Fixtures defined in `tests/conftest.py`
*   Uses pytest with asyncio support

```bash
# Run the tests
uv run pytest
# Run the tests with coverage report
uv run pytest -v --cov=zsim --cov-report=html
```

##  Further Development

Refer to the [Development Guide](https://github.com/ZZZSimulator/ZSim/wiki/%E8%B4%A1%E7%8C%AE%E6%8C%87%E5%8D%97-Develop-Guide) for detailed information on contributing to the project and the future development plans.