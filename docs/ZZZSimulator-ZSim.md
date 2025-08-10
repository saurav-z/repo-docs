# ZSim: Zenless Zone Zero Battle Simulator & Damage Calculator

**Unleash the power of optimized team compositions with ZSim, the ultimate battle simulator for Zenless Zone Zero!** ([View the original repository](https://github.com/ZZZSimulator/ZSim))

![ZSim Logo](docs/img/横板logo成图.png)

## Introduction

ZSim is a powerful battle simulator and damage calculator designed for Zenless Zone Zero (ZZZ), an action role-playing game by Hoyoverse. It allows you to analyze and optimize your team's damage output with ease. This tool is fully automated, eliminating the need for manual skill sequence input (though sequence mode can be incorporated). Simply equip your agents, select an Action Priority List (APL), and run the simulation to see the results!

## Key Features

*   **Automated Battle Simulation:** Simulates battles based on preset APLs, eliminating manual input.
*   **Comprehensive Damage Calculation:** Calculates total damage output based on team composition, taking character and equipment characteristics into account.
*   **Visual Reporting:** Generates detailed results in visual charts and tables for easy analysis.
*   **Agent Customization:**  Allows you to edit agent equipment and customize your strategies.
*   **APL Editing:** Enables you to modify APL code to fine-tune your team's actions.

## Installation

### Prerequisites: Install UV Package Manager

ZSim uses `uv` for dependency management.  Install it based on your operating system:

*   **Using pip (if you have Python):**
    ```bash
    pip install uv
    ```

*   **macOS or Linux:**
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

*   **Windows 11 (24H2 or later):**
    ```bash
    winget install --id=astral-sh.uv  -e
    ```

*   **Older Windows versions:**
    ```powershell
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    ```

*   **For more detailed installation instructions, please refer to the official UV documentation:**  [https://docs.astral.sh/uv/getting-started/installation/](https://docs.astral.sh/uv/getting-started/installation/)

### Install and Run ZSim

1.  **Download the latest source code:**  From the release page or clone the repository using `git clone`.
2.  **Navigate to the project directory** in your terminal.
3.  **Install dependencies and run the simulator:**
    ```bash
    uv sync
    uv run zsim run
    ```

## Development

### Key Components

ZSim's architecture comprises several key components:

1.  **Simulation Engine:** Located in `zsim/simulator/`, this handles the core battle simulation logic.
2.  **Web API:** A FastAPI-based REST API found in `zsim/api_src/`, providing programmatic access to the simulator.
3.  **Web UI:**  The Streamlit-based interface in `zsim/webui.py` and a new Vue.js + Electron desktop application in `electron-app/`
4.  **CLI:** Command-line interface available via `zsim/run.py`.
5.  **Database:** SQLite-based for storing character and enemy configurations.
6.  **Electron App:** A desktop application built with Vue.js and Electron that communicates with the FastAPI backend.

### Setup and Installation for Development

1.  **Install UV (if you haven't already - see Installation instructions above)**
2.  **Set up development environment:**
    ```bash
    uv sync
    # For WebUI development:
    uv run zsim run
    # For FastAPI backend:
    uv run zsim api
    # For Electron App development, also install Node.js dependencies
    cd electron-app
    corepack install
    pnpm install
    ```

### Testing Structure

ZSim employs a robust testing structure, with the following components:

*   **Unit Tests:** Located in the `tests/` directory.
*   **API Tests:** Found in `tests/api/`.
*   **Fixtures:** Defined in `tests/conftest.py`.
*   **Testing Framework:**  Utilizes pytest with asyncio support.

```bash
# Run all tests:
uv run pytest
# Run tests with coverage report:
uv run pytest -v --cov=zsim --cov-report=html
```

##  TODO LIST

Refer to the [develop guide](https://github.com/ZZZSimulator/ZSim/wiki/%E8%B4%B1%E7%8C%AE%E6%8C%87%E5%8D%97-Develop-Guide) for more details on the project's roadmap and contributing guidelines.