# ZSim: The Ultimate Zenless Zone Zero Battle Simulator & Damage Calculator

[View the original repository on GitHub](https://github.com/ZZZSimulator/ZSim)

![ZSim Logo](docs/img/横板logo成图.png)

## Introduction

Maximize your team's potential in Zenless Zone Zero with **ZSim**, a powerful and user-friendly battle simulator and damage calculator. ZSim automatically calculates total damage output, analyzes team compositions, and generates insightful reports, allowing you to optimize your strategies and achieve peak performance.

## Key Features

*   **Automated Simulation:** No manual skill sequences needed; ZSim automatically simulates battles based on Action Priority Lists (APLs).
*   **Comprehensive Damage Calculation:** Accurately calculates total damage based on character equipment and team composition.
*   **Visual Reports:** Generates detailed damage information through interactive charts and tables.
*   **Flexible Configuration:** Edit agent equipment and customize APL codes.
*   **Multiple Access Points:** Use the Web UI, CLI, or API to access ZSim's power.

## Installation

### Prerequisites
*   Python (if you want to develop)
*   uv (Python package manager)

### Install uv

Choose one method to install uv:

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

### Install and Run ZSim

1.  **Navigate:** Open your terminal in the project directory.
2.  **Sync Dependencies:**
    ```bash
    uv sync
    ```
3.  **Run ZSim:**
    ```bash
    uv run zsim run
    ```

## Development

### Key Components

*   **Simulation Engine:** Core logic in `zsim/simulator/`
*   **Web API:** FastAPI-based REST API in `zsim/api_src/`
*   **Web UI:** Streamlit-based interface in `zsim/webui.py` and a new Vue.js + Electron desktop application in `electron-app/`
*   **CLI:** Command-line interface via `zsim/run.py`
*   **Database:** SQLite-based storage for character/enemy configurations
*   **Electron App:** Desktop application built with Vue.js and Electron that communicates with the FastAPI backend

### Setup and Installation

```bash
# Install UV package manager first
uv sync
# For WebUI develop
uv run zsim run 
# For FastAPI backend
uv run zsim api

# For Electron App development, also install Node.js dependencies
cd electron-app
corepack install
pnpm install
```

### Testing Structure

*   Unit tests in `tests/` directory
*   API tests in `tests/api/`
*   Fixtures defined in `tests/conftest.py`
*   Uses pytest with asyncio support

```bash
# Run the tests
uv run pytest
# Run the tests with coverage report
uv run pytest -v --cov=zsim --cov-report=html
```

## Further Development

For more details, check the [develop guide](https://github.com/ZZZSimulator/ZSim/wiki/%E8%B4%A1%E7%8C%AE%E6%8C%87%E5%8D%97-Develop-Guide) for details on contributing.