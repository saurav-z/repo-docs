# ZSim: Zenless Zone Zero Damage Calculator and Battle Simulator

**Maximize your Zenless Zone Zero team's potential with ZSim, a powerful and user-friendly damage calculator and battle simulator!** 

[View the original repository on GitHub](https://github.com/ZZZSimulator/ZSim)

## Key Features

*   **Automatic Battle Simulation:**  No manual skill sequence setup required; ZSim automatically simulates battles based on a pre-set Action Priority List (APL).
*   **Comprehensive Damage Calculation:** Calculates total damage output for your team compositions, considering character equipment and weapon characteristics.
*   **Detailed Reporting:** Generates visual charts and tables for in-depth damage analysis, including individual character damage breakdowns.
*   **User-Friendly Interface:** Easy-to-use interface for editing agent equipment and APL codes.
*   **Flexible Team Building:**  Experiment with different team compositions to optimize your performance.

## Installation

ZSim is easy to set up.  Follow these steps:

1.  **Install UV Package Manager (if you haven't already):**

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
    *   **Older Windows versions:**
        ```powershell
        powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
        ```
    *   **For more detailed instructions, see the official UV documentation:** [https://docs.astral.sh/uv/getting-started/installation/](https://docs.astral.sh/uv/getting-started/installation/)

2.  **Install and Run ZSim:**

    *   Open your terminal in the project directory and run these commands:
        ```bash
        uv sync
        uv run zsim run
        ```

## Development

### Key Components

ZSim is built using a modular architecture:

1.  **Simulation Engine:** Core logic within the `zsim/simulator/` directory handles battle simulations.
2.  **Web API:**  A FastAPI-based REST API in `zsim/api_src/` provides programmatic access.
3.  **Web UI:**  A Streamlit-based interface in `zsim/webui.py` and a new Vue.js + Electron desktop application in `electron-app/` provide user interfaces.
4.  **CLI:**  Command-line interface available via `zsim/run.py`.
5.  **Database:** SQLite-based storage for character and enemy configurations.
6.  **Electron App:** Desktop application built with Vue.js and Electron that communicates with the FastAPI backend.

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

ZSim utilizes comprehensive testing:

*   Unit tests are located in the `tests/` directory.
*   API tests are in `tests/api/`.
*   Fixtures are defined in `tests/conftest.py`.
*   pytest with asyncio support is used for testing.

### Running Tests

```bash
# Run the tests
uv run pytest
# Run the tests with coverage report
uv run pytest -v --cov=zsim --cov-report=html
```

## TODO List

For more details on contributions and future development plans, consult the [Develop Guide](https://github.com/ZZZSimulator/ZSim/wiki/%E8%B4%A1%E7%8C%AE%E6%8C%87%E5%8D%97-Develop-Guide).