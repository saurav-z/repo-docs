# ZSim: Zenless Zone Zero Battle Simulator & Damage Calculator 🚀

**Maximize your Zenless Zone Zero team's potential with ZSim, a powerful, automated battle simulator and damage calculator.** ([View the Original Repository](https://github.com/ZZZSimulator/ZSim))

![ZSim Logo](docs/img/横板logo成图.png)

## About ZSim

ZSim is an advanced battle simulator and damage calculator designed specifically for Zenless Zone Zero (ZZZ), the action RPG from Hoyoverse.  It takes the guesswork out of team building by automatically simulating battles based on your team's equipment and a predefined Action Priority List (APL), providing detailed damage output analysis.

## Key Features

*   **Automated Battle Simulation:**  Simulates battles automatically based on the chosen APL.
*   **Comprehensive Damage Calculation:** Calculates total damage output, considering character weapons and equipment.
*   **Visual Reports:** Generates detailed damage information in easy-to-understand visual charts and tables.
*   **Character Equipment Customization:**  Allows you to edit and optimize your agents' equipment.
*   **APL Customization:**  Offers the ability to modify APL code for fine-grained control (if needed).

## Getting Started

### Installation

1.  **Download:** Download the latest source code from the [releases page](link to releases page, not available in the original README) or clone the repository using Git:

    ```bash
    git clone https://github.com/ZZZSimulator/ZSim.git
    ```

2.  **Install UV (Package Manager):** ZSim utilizes `uv` for package management. If you don't have it, install it using one of the following methods:

    ```bash
    # Using pip if you have python installed:
    pip install uv
    ```

    ```bash
    # On macOS or Linux:
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

    ```bash
    # On Windows 11 24H2 or later:
    winget install --id=astral-sh.uv  -e
    ```

    ```bash
    # On lower version of Windows:
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    ```

    For more installation details, refer to the official `uv` installation guide: [https://docs.astral.sh/uv/getting-started/installation/](https://docs.astral.sh/uv/getting-started/installation/)

3.  **Run ZSim:** Navigate to the project directory in your terminal and run the following commands:

    ```bash
    uv sync
    uv run zsim run
    ```

## Development

### Core Components

*   **Simulation Engine:**  `zsim/simulator/` - The core logic for the battle simulation.
*   **Web API:** `zsim/api_src/` - A FastAPI-based REST API for programmatic access.
*   **Web UI:** `zsim/webui.py` (Streamlit) and `electron-app/` (Vue.js + Electron) - User interfaces for interaction.
*   **CLI:** `zsim/run.py` - Command-line interface.
*   **Database:** SQLite-based storage for character and enemy configurations.
*   **Electron App:**  A desktop application built with Vue.js and Electron, communicating with the FastAPI backend.

### Setup and Installation for Development

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

## Contributing

Please check the [develop guide](https://github.com/ZZZSimulator/ZSim/wiki/%E8%B4%A1%E7%8C%AE%E6%8C%87%E5%8D%97-Develop-Guide) for contribution guidelines.