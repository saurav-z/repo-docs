# ZSim: Zenless Zone Zero Battle Simulator & Damage Calculator

**Maximize your team's performance in Zenless Zone Zero with ZSim, the automated battle simulator that empowers you to strategize and optimize your gameplay.**  [View the project on GitHub](https://github.com/ZZZSimulator/ZSim)

![ZSim Logo](docs/img/横板logo成图.png)

## Overview

ZSim is a powerful battle simulator and damage calculator specifically designed for Zenless Zone Zero (ZZZ), the action role-playing game from Hoyoverse.  It allows you to analyze team compositions, experiment with equipment builds, and understand the damage output of your agents.  ZSim automates the simulation process, eliminating the need for manual skill sequence input.  Simply configure your agents' equipment and select an Action Priority List (APL) to receive comprehensive damage reports, visualized in charts and tables.

## Key Features

*   **Automated Battle Simulation:**  Simulates battles automatically based on preset APLs.
*   **Total Damage Calculation:** Accurately calculates total damage output for your team composition.
*   **Visual Data Reports:** Generates informative charts and tables for easy analysis.
*   **Detailed Damage Breakdown:** Provides granular damage information for each character.
*   **Agent Equipment Editor:** Allows you to customize and configure agent equipment.
*   **APL Code Editing:**  Customize your action priority lists for tailored simulations.

## Installation

### Prerequisites:

*   **Python:** Ensure you have Python installed.

### Install UV Package Manager (If not already installed)

ZSim uses the `uv` package manager. If you don't have it, install it using one of the following methods:

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

1.  **Clone the repository:** Download the latest source code from the release page or use `git clone`.
2.  **Navigate to the project directory:** Open a terminal in the project's root directory.
3.  **Install dependencies and run:**
    ```bash
    uv sync
    uv run zsim run
    ```

## Development

### Key Components

*   **Simulation Engine:** Core logic in `zsim/simulator/` for battle simulations.
*   **Web API:** FastAPI-based REST API in `zsim/api_src/` for programmatic access.
*   **Web UI:** Streamlit-based interface in `zsim/webui.py` and a new Vue.js + Electron desktop application in `electron-app/`.
*   **CLI:** Command-line interface via `zsim/run.py`.
*   **Database:** SQLite-based storage for character/enemy configurations.
*   **Electron App:** Desktop application built with Vue.js and Electron that communicates with the FastAPI backend.

### Setup and Installation (Development)

```bash
# Install UV package manager first
uv sync

# For WebUI development
uv run zsim run 

# For FastAPI backend
uv run zsim api

# For Electron App development, also install Node.js dependencies
cd electron-app
yarn install
```

### Testing Structure

*   **Unit Tests:** Located in the `tests/` directory.
*   **API Tests:** Located in the `tests/api/` directory.
*   **Fixtures:** Defined in `tests/conftest.py`.
*   **Testing Framework:** Uses pytest with asyncio support.

```bash
# Run the tests
uv run pytest
# Run the tests with coverage report
uv run pytest -v --cov=zsim --cov-report=html
```

## Further Information

For detailed information on contributing, see the [development guide](https://github.com/ZZZSimulator/ZSim/wiki/%E8%B4%A1%E7%8C%AE%E6%8C%87%E5%8D%97-Develop-Guide).