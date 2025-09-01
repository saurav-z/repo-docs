# ZSim: Zenless Zone Zero Battle Simulator & Damage Calculator

**Maximize your team's potential in Zenless Zone Zero with ZSim, a powerful, automated battle simulator and damage calculator.** (Original Repo: [https://github.com/ZZZSimulator/ZSim](https://github.com/ZZZSimulator/ZSim))

## Key Features

*   **Automated Battle Simulation:** Automatically simulates battles based on Action Priority Lists (APLs), eliminating the need for manual skill sequence input.
*   **Comprehensive Damage Calculation:** Accurately calculates total damage output for your team compositions, considering character, weapon and equipment stats.
*   **User-Friendly Interface:** Features an intuitive interface for editing agent equipment and APL code.
*   **Visual Data Analysis:** Generates informative charts and tables to visualize damage output and character performance.
*   **Detailed Character Information:** Provides in-depth damage breakdowns for each character in your team.

## Installation

### Prerequisites

*   **UV Package Manager:** Install UV using one of the following methods:

    *   **Using pip (if Python installed):** `pip install uv`
    *   **macOS or Linux:** `curl -LsSf https://astral.sh/uv/install.sh | sh`
    *   **Windows 11 24H2 or later:** `winget install --id=astral-sh.uv  -e`
    *   **Older Windows versions:** `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`

    For detailed installation instructions, see the official UV documentation: [https://docs.astral.sh/uv/getting-started/installation/](https://docs.astral.sh/uv/getting-started/installation/)

### Installing and Running ZSim

1.  **Download:** Download the latest source code from the [releases page](https://github.com/ZZZSimulator/ZSim/releases) or use `git clone`.
2.  **Navigate to the project directory:** Open a terminal in the ZSim project directory.
3.  **Install Dependencies:** `uv sync`
4.  **Run the Simulator:**

    *   **Web UI:** `uv run zsim run`
    *   **FastAPI Backend:** `uv run zsim api`
    *   **Electron App (Development):**
        ```bash
        cd electron-app
        yarn install
        ```

## Development

### Core Components

*   **Simulation Engine:** Core logic in `zsim/simulator/`.
*   **Web API:** FastAPI-based REST API in `zsim/api_src/`.
*   **Web UI:** Streamlit-based interface in `zsim/webui.py` and a new Vue.js + Electron desktop application in `electron-app/`.
*   **CLI:** Command-line interface via `zsim/run.py`.
*   **Database:** SQLite-based storage for character/enemy configurations.
*   **Electron App:** Desktop application built with Vue.js and Electron that communicates with the FastAPI backend.

### Testing

*   **Unit Tests:** Located in the `tests/` directory.
*   **API Tests:** Located in `tests/api/`.
*   **Fixtures:** Defined in `tests/conftest.py`.
*   **Testing Framework:** pytest with asyncio support.

#### Running Tests

*   `uv run pytest` (Runs all tests)
*   `uv run pytest -v --cov=zsim --cov-report=html` (Runs tests with coverage report)

## Further Information

*   For development details, see the [develop guide](https://github.com/ZZZSimulator/ZSim/wiki/%E8%B4%A1%E7%8C%AE%E6%8C%87%E5%8D%97-Develop-Guide).