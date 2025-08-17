# ZSim: Your Ultimate Zenless Zone Zero Battle Simulator

**Unleash the power of strategic team building and damage analysis with ZSim, your go-to simulator for Zenless Zone Zero (ZZZ)!**  ([View the Original Repo](https://github.com/ZZZSimulator/ZSim))

![ZSim Logo](docs/img/横板logo成图.png)

## Introduction to ZSim

ZSim is a powerful battle simulator and damage calculator meticulously designed for Zenless Zone Zero, the thrilling action game from Hoyoverse.  It allows you to analyze team compositions and optimize your agent builds for maximum damage output.  Forget tedious manual skill sequencing; ZSim's **fully automated** simulations take the guesswork out of your strategy, providing clear insights into your team's performance.

## Key Features

*   **Automatic Battle Simulation:**  No need to manually input skill sequences - ZSim handles the action, triggering buffs and debuffs, and analyzing the results based on your chosen Action Priority List (APL).
*   **Comprehensive Damage Calculation:**  Calculate total damage output for your entire team, considering each agent's weapon, equipment, and skills.
*   **Visual Data Reporting:** Generate insightful charts and tables to visualize your team's performance, making it easy to understand and compare different builds.
*   **Agent Equipment Customization:**  Easily edit your agents' equipment to experiment with various builds and see how they impact your damage output.
*   **APL Editing:** Fine-tune your team's action priority with customizable APLs to optimize your strategy.

## Installation

Get started with ZSim in a few simple steps:

1.  **Download ZSim:** Download the latest source code from the [releases page](link to releases page, if available) or use `git clone`.
2.  **Install UV (if you haven't already):** UV is a package manager required to run the project.  Follow the instructions below based on your operating system:

    *   **Using pip (if you have Python installed):**
        ```bash
        pip install uv
        ```
    *   **On macOS or Linux:**
        ```bash
        curl -LsSf https://astral.sh/uv/install.sh | sh
        ```
    *   **On Windows 11 24H2 or later:**
        ```bash
        winget install --id=astral-sh.uv  -e
        ```
    *   **On older versions of Windows:**
        ```powershell
        powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
        ```
    *   **Official UV Installation Guide:** [https://docs.astral.sh/uv/getting-started/installation/](https://docs.astral.sh/uv/getting-started/installation/)
3.  **Install and Run ZSim:**  Open your terminal in the project directory and execute the following commands:

    ```bash
    uv sync
    uv run zsim run
    ```

## Development

### Key Components

*   **Simulation Engine:** Core logic for battle simulation located in `zsim/simulator/`.
*   **Web API:** FastAPI-based REST API in `zsim/api_src/` for programmatic access.
*   **Web UI:**  Streamlit-based interface in `zsim/webui.py` and a new Vue.js + Electron desktop application in `electron-app/`.
*   **CLI:** Command-line interface via `zsim/run.py`.
*   **Database:** SQLite-based storage for character/enemy configurations.
*   **Electron App:** Desktop application built with Vue.js and Electron that communicates with the FastAPI backend.

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

*   Unit tests in the `tests/` directory.
*   API tests in `tests/api/`.
*   Fixtures defined in `tests/conftest.py`.
*   Uses pytest with asyncio support.

```bash
# Run the tests
uv run pytest
# Run the tests with coverage report
uv run pytest -v --cov=zsim --cov-report=html
```

## TODO List

For more details on contributing and future development, please refer to the [develop guide](https://github.com/ZZZSimulator/ZSim/wiki/%E8%B4%A1%E7%8C%AE%E6%8C%87%E5%8D%97-Develop-Guide).