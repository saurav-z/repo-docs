# ZSim: Your Ultimate Zenless Zone Zero Battle Simulator

**Maximize your team's potential in Zenless Zone Zero with ZSim, a powerful and user-friendly battle simulator.** ([Original Repo](https://github.com/ZZZSimulator/ZSim))

![ZSim Logo](docs/img/横板logo成图.png)

## Introduction

ZSim is a comprehensive battle simulator and damage calculator specifically designed for Zenless Zone Zero (ZZZ), Hoyoverse's action RPG. This tool allows you to optimize your team compositions and equipment setups for maximum damage output.  It **automatically simulates** battles based on Action Priority Lists (APLs), providing detailed insights into your team's performance without the need for manual skill sequence input.

## Key Features

*   **Automated Battle Simulation:**  Simulates battles based on your chosen APLs, eliminating manual input.
*   **Damage Calculation:** Accurately calculates total damage output, considering character stats, equipment, and weapon characteristics.
*   **Visual Reports:** Generates easy-to-understand charts and tables to visualize battle results and damage breakdowns.
*   **Agent Equipment Editing:** Customize your agents' equipment within the simulator.
*   **APL Customization:**  Edit and fine-tune Action Priority Lists (APLs) to optimize your team's actions.
*   **Detailed Damage Information:** Provides a comprehensive breakdown of damage for each character in your team.

## Installation

To get started with ZSim, follow these simple steps:

1.  **Install UV (if you haven't already):**  UV is a fast Python package manager that is required.

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

    For more detailed UV installation instructions, visit the official documentation: [https://docs.astral.sh/uv/getting-started/installation/](https://docs.astral.sh/uv/getting-started/installation/)

2.  **Install and Run ZSim:** Open your terminal in the project directory and run the following commands:

    ```bash
    uv sync
    uv run zsim run
    ```

## Development

### Key Components

*   **Simulation Engine:** Located in `zsim/simulator/`, responsible for the core battle simulation logic.
*   **Web API:** A FastAPI-based REST API in `zsim/api_src/` for programmatic interaction.
*   **Web UI:** A Streamlit-based interface in `zsim/webui.py`, with a new Vue.js + Electron desktop application in `electron-app/`.
*   **CLI:** Command-line interface accessible via `zsim/run.py`.
*   **Database:** Uses SQLite for storing character and enemy configurations.
*   **Electron App:** A desktop application built with Vue.js and Electron, communicating with the FastAPI backend.

### Setup and Installation for Development

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

ZSim uses pytest for testing, with the following structure:

*   Unit tests are located in the `tests/` directory.
*   API tests are located in `tests/api/`.
*   Fixtures are defined in `tests/conftest.py`.

To run tests:

```bash
# Run the tests
uv run pytest
# Run the tests with coverage report
uv run pytest -v --cov=zsim --cov-report=html
```

## Further Information

For detailed development information and contribution guidelines, please refer to the [development guide](https://github.com/ZZZSimulator/ZSim/wiki/%E8%B4%A1%E7%8C%AE%E6%8C%87%E5%8D%97-Develop-Guide).