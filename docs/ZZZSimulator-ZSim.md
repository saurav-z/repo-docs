# ZSim: Zenless Zone Zero Battle Simulator & Damage Calculator

**Maximize your team's potential with ZSim, the ultimate battle simulator for Zenless Zone Zero (ZZZ)!**

[View the ZSim Repository on GitHub](https://github.com/ZZZSimulator/ZSim)

## Key Features

*   **Automatic Battle Simulation:** Automatically simulates battles based on Action Priority Lists (APLs), eliminating the need for manual skill sequence input.
*   **Comprehensive Damage Calculation:** Accurately calculates total damage output, considering character equipment and weapon characteristics.
*   **User-Friendly Interface:** Provides an intuitive interface for easy team composition evaluation.
*   **Visual Reports:** Generates informative visual charts and tables to analyze battle results.
*   **Equipment Customization:** Allows you to edit and optimize agent equipment.
*   **APL Customization:** Enables the editing of APL code for advanced strategy.

## Installation

### Prerequisites

1.  **Install `uv` (if you haven't already):**  `uv` is a fast package manager used for managing project dependencies. Choose one of the following installation methods based on your operating system:

    *   **Using `pip` (if you have Python installed):**
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

    *For a detailed installation guide, see: [uv Installation Guide](https://docs.astral.sh/uv/getting-started/installation/)*

### Installing and Running ZSim

1.  **Download:** Get the latest source code from the [releases page](https://github.com/ZZZSimulator/ZSim/releases) or clone the repository using `git clone`.
2.  **Navigate:** Open a terminal and navigate to the project's directory.
3.  **Install Dependencies and Run:** Execute the following commands:

    ```bash
    uv sync
    uv run zsim run
    ```

## Development

### Key Components

*   **Simulation Engine:** Core logic for battle simulation (`zsim/simulator/`).
*   **Web API:** FastAPI-based REST API for programmatic access (`zsim/api_src/`).
*   **Web UI:** Streamlit-based interface (`zsim/webui.py`) and a new Vue.js + Electron desktop application (`electron-app/`).
*   **CLI:** Command-line interface (`zsim/run.py`).
*   **Database:** SQLite-based storage for character and enemy configurations.
*   **Electron App:** Desktop application built with Vue.js and Electron.

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

*   **Unit Tests:** Located in the `tests/` directory.
*   **API Tests:** Located in the `tests/api/` directory.
*   **Fixtures:** Defined in `tests/conftest.py`.
*   **Testing Framework:** Uses `pytest` with asyncio support.

### Running Tests

```bash
# Run the tests
uv run pytest
# Run the tests with coverage report
uv run pytest -v --cov=zsim --cov-report=html
```

## Further Information

*   For more detailed information, check out the [develop guide](https://github.com/ZZZSimulator/ZSim/wiki/%E8%B4%A1%E7%8C%AE%E6%8C%87%E5%8D%97-Develop-Guide).