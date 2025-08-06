# ZSim: Zenless Zone Zero Battle Simulator & Damage Calculator

Calculate the optimal team damage output in Zenless Zone Zero with ZSim, a powerful and easy-to-use simulator.  [View the original repository](https://github.com/ZZZSimulator/ZSim).

![ZSim Logo](docs/img/横板logo成图.png)

## Key Features

*   **Automated Battle Simulation:** Automatically simulates battles based on your team's actions.
*   **Damage Calculation:** Accurately calculates total damage output for your team.
*   **User-Friendly Interface:**  Provides an intuitive interface for setting up teams and running simulations.
*   **Visual Reports:** Generates charts and tables for easy analysis of damage results.
*   **Character & Equipment Customization:** Edit agent equipment to optimize your team's performance.
*   **Action Priority List (APL) Support:**  Utilizes APLs to simulate team actions and buffs.
*   **Detailed Damage Information:** Provides in-depth damage breakdowns for each character.

## Installation

### Prerequisites

*   **Python:** Ensure you have Python installed on your system.
*   **UV Package Manager:** ZSim utilizes the `uv` package manager for dependency management. Install it by following the instructions below:

    ```bash
    # Using pip (if you have python installed):
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

    Or check the official installation guide: [https://docs.astral.sh/uv/getting-started/installation/](https://docs.astral.sh/uv/getting-started/installation/)

### Installing and Running ZSim

1.  **Download the Source Code:** Get the latest version from the [releases page](https://github.com/ZZZSimulator/ZSim/releases) or clone the repository using `git clone`.
2.  **Navigate to the Project Directory:** Open your terminal and navigate to the directory where you downloaded ZSim.
3.  **Install Dependencies and Run:** Execute the following commands:

    ```bash
    uv sync
    uv run zsim run
    ```

## Development

### Key Components

*   **Simulation Engine:** Located in `zsim/simulator/` (core battle logic).
*   **Web API:** Built with FastAPI in `zsim/api_src/` (REST API).
*   **Web UI:** Uses Streamlit (`zsim/webui.py`) and a Vue.js + Electron desktop app (`electron-app/`).
*   **CLI:**  Command-line interface in `zsim/run.py`.
*   **Database:** Uses SQLite for storing character and enemy configurations.
*   **Electron App:** Desktop application (Vue.js and Electron) communicating with the FastAPI backend.

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
corepack install
pnpm install
```

### Testing Structure

*   Unit tests are in the `tests/` directory.
*   API tests are in `tests/api/`.
*   Fixtures are defined in `tests/conftest.py`.
*   Uses pytest with asyncio support.

```bash
# Run the tests
uv run pytest
# Run the tests with coverage report
uv run pytest -v --cov=zsim --cov-report=html
```

## TODO List

Refer to the [development guide](https://github.com/ZZZSimulator/ZSim/wiki/%E8%B4%A1%E7%8C%AE%E6%8C%87%E5%8D%97-Develop-Guide) for details.