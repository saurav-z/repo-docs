# ZSim: Zenless Zone Zero Battle Simulator & Damage Calculator

**Unleash the power of your team with ZSim, the ultimate battle simulator and damage calculator for Zenless Zone Zero.** ([Original Repository](https://github.com/ZZZSimulator/ZSim))

![ZSim Logo](docs/img/横板logo成图.png)

## What is ZSim?

ZSim is a powerful tool designed to simulate battles and calculate damage output in Zenless Zone Zero (ZZZ). It automatically simulates combat based on your team composition, equipment, and Action Priority List (APL), providing detailed insights into your team's performance.

## Key Features

*   **Automatic Battle Simulation:** No manual skill sequence input needed. ZSim intelligently simulates combat.
*   **Comprehensive Damage Calculation:** Calculates total damage output based on team composition, agent equipment, and weapons.
*   **Visual Reporting:** Generates visual charts and tables for easy analysis of results.
*   **Agent Customization:** Allows you to edit agent equipment to optimize your builds.
*   **APL Customization:** Edit the Action Priority List (APL) to fine-tune your team's strategy.
*   **User-Friendly Interface:** Provides an intuitive interface for easy navigation and analysis.

## Installation

### Prerequisites: Install UV (if you haven't already)

Choose your operating system:

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

1.  Download the latest source code from the [release page](link to releases if available) or use `git clone`.
2.  Open a terminal in the project directory.
3.  Run the following commands:

```bash
uv sync
uv run zsim run
```

## Development

### Key Components

*   **Simulation Engine:** Core logic in `zsim/simulator/`.
*   **Web API:** FastAPI-based REST API in `zsim/api_src/`.
*   **Web UI:** Streamlit-based interface in `zsim/webui.py` and new Vue.js + Electron desktop application in `electron-app/`.
*   **CLI:** Command-line interface via `zsim/run.py`.
*   **Database:** SQLite-based storage for character/enemy configurations.
*   **Electron App:** Desktop application built with Vue.js and Electron.

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
yarn install
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

## TODO

Check the [develop guide](https://github.com/ZZZSimulator/ZSim/wiki/%E8%B4%A1%E7%8C%AE%E6%8C%87%E5%8D%97-Develop-Guide) for more information on development tasks and contributions.