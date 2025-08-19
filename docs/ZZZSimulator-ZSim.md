# ZSim: Zenless Zone Zero Damage Calculator and Battle Simulator

**Maximize your Zenless Zone Zero team's potential with ZSim, the automated damage calculator and battle simulator!**

[View the original repository on GitHub](https://github.com/ZZZSimulator/ZSim)

## About ZSim

ZSim is a powerful tool for Zenless Zone Zero (ZZZ) players, offering a comprehensive battle simulation and damage calculation experience.  It automates the process of evaluating team compositions and equipment, providing valuable insights to optimize your gameplay. Simply equip your agents, select an Action Priority List (APL), and run the simulation to see detailed damage reports.

## Key Features

*   **Automated Battle Simulation:**  Simulates battles based on preset APLs, eliminating the need for manual skill sequence input.
*   **Damage Calculation:**  Accurately calculates total damage output for a team, considering character stats, weapons, and equipment.
*   **Visual Reports:** Generates easy-to-understand charts and tables for in-depth damage analysis.
*   **Character Equipment Customization:**  Allows you to edit and customize agent equipment to test various builds.
*   **APL Editing:**  Offers the ability to modify APL code for advanced simulation control.

## Installation

### Prerequisites

You'll need the `uv` package manager. Install it using one of the following methods:

*   **Using `pip` (if Python is installed):**
    ```bash
    pip install uv
    ```
*   **macOS or Linux:**
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```
*   **Windows 11 24H2 or later:**
    ```bash
    winget install --id=astral-sh.uv  -e
    ```
*   **Older Windows Versions:**
    ```powershell
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    ```
    Check the official installation guide for more details: [https://docs.astral.sh/uv/getting-started/installation/](https://docs.astral.sh/uv/getting-started/installation/)

### ZSim Installation and Run

1.  **Clone or Download:** Download the source code or clone the repository using `git clone`.
2.  **Navigate to Project Directory:** Open a terminal in the project's root directory.
3.  **Sync Dependencies:**
    ```bash
    uv sync
    ```
4.  **Run ZSim:**
    ```bash
    uv run zsim run
    ```

## Development

### Key Components

*   **Simulation Engine:** Core battle simulation logic (`zsim/simulator/`).
*   **Web API:** REST API built with FastAPI (`zsim/api_src/`).
*   **Web UI:** Streamlit-based interface (`zsim/webui.py`) and a new Vue.js + Electron desktop application (`electron-app/`).
*   **CLI:** Command-line interface (`zsim/run.py`).
*   **Database:** SQLite database for character and enemy configurations.
*   **Electron App:** Desktop application using Vue.js and Electron, communicating with the FastAPI backend.

### Setup

```bash
# Install UV package manager first (if not already)
uv sync

# For WebUI development
uv run zsim run

# For FastAPI backend
uv run zsim api

# For Electron App development, also install Node.js dependencies
cd electron-app
yarn install
```

### Testing

-   Unit tests are located in the `tests/` directory.
-   API tests are located in `tests/api/`.
-   Fixtures are defined in `tests/conftest.py`.
-   Uses pytest with asyncio support.

```bash
# Run tests
uv run pytest

# Run tests with coverage report
uv run pytest -v --cov=zsim --cov-report=html
```

## Further Development

For more details on contributions, check out the [Develop Guide](https://github.com/ZZZSimulator/ZSim/wiki/%E8%B4%A1%E7%8C%AE%E6%8C%87%E5%8D%97-Develop-Guide) on the wiki.