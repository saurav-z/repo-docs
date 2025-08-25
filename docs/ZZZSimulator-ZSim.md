# ZSim: Zenless Zone Zero Battle Simulator & Damage Calculator

**Maximize your team's potential in Zenless Zone Zero with ZSim, the ultimate damage calculator and battle simulator!** ([View the original repository](https://github.com/ZZZSimulator/ZSim))

![ZSim Logo](./docs/img/横板logo成图.png)

## About ZSim

ZSim is a powerful, automated battle simulator and damage calculator specifically designed for Zenless Zone Zero (ZZZ), the action game from Hoyoverse.  It allows you to analyze team compositions and optimize your builds without manual skill sequence input.  Simply equip your agents, select an Action Priority List (APL), and run the simulation to generate detailed reports and visualizations of your team's damage output.

## Key Features

*   **Automated Battle Simulation:**  No need for manual input of skill sequences.
*   **Comprehensive Damage Calculation:**  Accurately calculates total damage output based on team composition, character equipment, and weapon characteristics.
*   **Visualized Results:** Generates clear and informative charts and tables for easy analysis.
*   **Character Equipment Editing:** Customize your agents' equipment to fine-tune your builds.
*   **APL Customization:**  Edit Action Priority Lists (APLs) to tailor your team's actions.
*   **User-Friendly Interface:** Easy-to-use interface for efficient simulation and analysis.

## Installation

### Prerequisites: Install UV (Package Manager)

You'll need the `uv` package manager. Follow the instructions below based on your operating system:

*   **Using pip (if Python is installed):**

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

    For more details, see the official installation guide: [https://docs.astral.sh/uv/getting-started/installation/](https://docs.astral.sh/uv/getting-started/installation/)

### Install and Run ZSim

1.  Open your terminal in the project directory.
2.  Install dependencies:

    ```bash
    uv sync
    ```
3.  Run ZSim:

    ```bash
    uv run zsim run
    ```

## Development

### Key Components

*   **Simulation Engine:**  Located in `zsim/simulator/` - handles the core battle logic.
*   **Web API:** A FastAPI-based REST API in `zsim/api_src/`.
*   **Web UI:**  Streamlit-based interface in `zsim/webui.py` and a new Vue.js + Electron desktop application (`electron-app/`).
*   **CLI:** Command-line interface via `zsim/run.py`.
*   **Database:** SQLite database for storing character and enemy configurations.
*   **Electron App:** Desktop application developed with Vue.js and Electron, communicating with the FastAPI backend.

### Development Setup

```bash
# Install UV package manager first:
uv sync

# For WebUI development:
uv run zsim run

# For FastAPI backend:
uv run zsim api

# For Electron App development, you also need to install Node.js dependencies:
cd electron-app
yarn install
```

### Testing

*   **Unit tests:** Located in `tests/`
*   **API tests:** Located in `tests/api/`
*   **Fixtures:** Defined in `tests/conftest.py`
*   Uses pytest with asyncio support.

**Running Tests:**

```bash
# Run all tests:
uv run pytest

# Run tests with coverage report:
uv run pytest -v --cov=zsim --cov-report=html
```

## TODO List

Refer to the [Development Guide](https://github.com/ZZZSimulator/ZSim/wiki/%E8%B4%A1%E7%8C%AE%E6%8C%87%E5%8D%97-Develop-Guide) for detailed information.