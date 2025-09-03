# ZSim: Zenless Zone Zero Battle Simulator & Damage Calculator

**Unleash the power of your team with ZSim, the ultimate battle simulator for Zenless Zone Zero, providing in-depth damage analysis and optimized team compositions!**

[View the original repository on GitHub](https://github.com/ZZZSimulator/ZSim)

## Key Features

*   **Automatic Battle Simulation:**  Eliminates the need for manual skill sequence input by simulating battles based on Action Priority Lists (APLs).
*   **Comprehensive Damage Calculation:** Accurately calculates total damage output, considering character, weapon, and equipment stats.
*   **Visual Reports:** Generates interactive charts and tables to visualize damage data for easy analysis.
*   **Character & Equipment Customization:** Edit agent equipment to test different builds and strategies.
*   **APL Editing:** Customize Action Priority Lists to fine-tune your team's performance.

## Installation

### Prerequisites: Install UV Package Manager

ZSim utilizes the UV package manager for dependency management.  Install it using one of the following methods:

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

Or check the official installation guide: [https://docs.astral.sh/uv/getting-started/installation/](https://docs.astral.sh/uv/getting-started/installation/)

### Installing and Running ZSim

1.  **Download:** Download the latest source code from the [releases page](<URL>) or clone the repository using Git.
2.  **Navigate:** Open a terminal in the project directory.
3.  **Install Dependencies & Run:**

    ```bash
    uv sync
    uv run zsim run
    ```

## Development

### Core Components

ZSim is built using a modular architecture:

1.  **Simulation Engine:** (`zsim/simulator/`) Handles core battle simulation logic.
2.  **Web API:**  FastAPI-based REST API for programmatic access (`zsim/api_src/`).
3.  **Web UI:** Streamlit-based interface (`zsim/webui.py`) and a new Vue.js + Electron desktop application (`electron-app/`).
4.  **CLI:** Command-line interface (`zsim/run.py`).
5.  **Database:** SQLite-based storage for character and enemy configurations.
6.  **Electron App:** Desktop application (Vue.js & Electron) communicating with the FastAPI backend.

### Development Setup

1.  **Install Dependencies:** Ensure UV is installed (see above).
2.  **Web UI Development:**

    ```bash
    uv run zsim run
    ```

3.  **FastAPI Backend Development:**

    ```bash
    uv run zsim api
    ```

4.  **Electron App Development:** (also requires Node.js and Yarn)

    ```bash
    cd electron-app
    yarn install
    ```

### Testing

ZSim includes a comprehensive testing suite:

*   **Unit Tests:**  `tests/`
*   **API Tests:** `tests/api/`
*   **Fixtures:** Defined in `tests/conftest.py`
*   **Testing Framework:** Uses pytest with asyncio support.

To run the tests:

```bash
uv run pytest
```

To run tests with a coverage report:

```bash
uv run pytest -v --cov=zsim --cov-report=html
```

## Contributing

See the [Development Guide](https://github.com/ZZZSimulator/ZSim/wiki/%E8%B4%A1%E7%8C%AE%E6%8C%87%E5%8D%97-Develop-Guide) for detailed information on contributing to ZSim.