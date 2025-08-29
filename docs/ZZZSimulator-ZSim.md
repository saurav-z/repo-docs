# ZSim: Zenless Zone Zero Battle Simulator & Damage Calculator

**Maximize your team's potential in Zenless Zone Zero with ZSim, the automated battle simulator that provides in-depth damage analysis and visual reports.** (Link to original repo: [https://github.com/ZZZSimulator/ZSim](https://github.com/ZZZSimulator/ZSim))

![ZSim Logo](./docs/img/横板logo成图.png)

## About ZSim

ZSim is an advanced battle simulator and damage calculator specifically designed for the action-packed world of Zenless Zone Zero (ZZZ) from Hoyoverse. This tool empowers players to optimize their team compositions and equipment builds through detailed simulations and insightful data analysis. Unlike manual calculators, ZSim automates the entire process, eliminating the need for tedious skill sequence input and providing a user-friendly experience.

## Key Features

*   **Automated Battle Simulation:** Simulates battles based on your selected Action Priority List (APL) without manual input.
*   **Comprehensive Damage Calculation:** Calculates total damage output, considering agent equipment, weapons, and team synergies.
*   **Visual Reports & Analysis:** Generates visual charts and tables to provide detailed damage information for each character, making it easy to understand your team's performance.
*   **Equipment Customization:** Allows you to edit and customize agent equipment to test various builds.
*   **APL Editing:** Provides the ability to modify the APL to experiment with different team strategies.

## Installation Guide

ZSim is easy to set up. Follow these steps:

### 1.  Install UV Package Manager (if you haven't already)

Choose the installation method appropriate for your system:

```bash
# Using pip (if you have Python installed):
pip install uv
```

```bash
# On macOS or Linux:
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```bash
# On Windows 11 24H2 or later:
winget install --id=astral-sh.uv -e
```

```bash
# On lower versions of Windows:
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Alternatively, consult the official UV installation guide: [https://docs.astral.sh/uv/getting-started/installation/](https://docs.astral.sh/uv/getting-started/installation/)

### 2. Install and Run ZSim

1.  **Download:** Download the latest source code from the [releases page](https://github.com/ZZZSimulator/ZSim/releases) or clone the repository using Git.
2.  **Navigate:** Open your terminal and navigate to the project's directory.
3.  **Sync Dependencies:**

    ```bash
    uv sync
    ```
4.  **Run ZSim:**

    ```bash
    uv run zsim run
    ```

## Development Information

### Core Components

*   **Simulation Engine:** Core battle simulation logic located in `zsim/simulator/`.
*   **Web API:** REST API built with FastAPI, found in `zsim/api_src/`.
*   **Web UI:** Streamlit-based interface (`zsim/webui.py`) and a new Vue.js + Electron desktop application (`electron-app/`).
*   **CLI:** Command-line interface via `zsim/run.py`.
*   **Database:** SQLite database for storing character and enemy configurations.
*   **Electron App:** Desktop application built with Vue.js and Electron that interacts with the FastAPI backend.

### Setting up your Development Environment

1.  **Install Dependencies:**

    ```bash
    # Install UV package manager first
    uv sync
    ```

    For WebUI development:

    ```bash
    uv run zsim run
    ```

    For FastAPI backend development:

    ```bash
    uv run zsim api
    ```

    For Electron App development, install Node.js dependencies:

    ```bash
    cd electron-app
    yarn install
    ```

### Testing

ZSim uses pytest with asyncio support.

*   **Unit Tests:** Located in the `tests/` directory.
*   **API Tests:** Found in `tests/api/`.
*   **Fixtures:** Defined in `tests/conftest.py`.

To run tests:

```bash
# Run tests
uv run pytest
# Run tests with coverage report
uv run pytest -v --cov=zsim --cov-report=html
```

## Further Information

Explore the [develop guide](https://github.com/ZZZSimulator/ZSim/wiki/%E8%B4%A1%E7%8C%AE%E6%8C%87%E5%8D%97-Develop-Guide) for comprehensive development details and contribution guidelines.