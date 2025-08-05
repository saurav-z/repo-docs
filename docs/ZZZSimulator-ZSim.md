# ZSim: Zenless Zone Zero Battle Simulator & Damage Calculator

Tired of manually calculating damage in Zenless Zone Zero? ZSim is a powerful, **fully automated** battle simulator and damage calculator that lets you optimize your team compositions and strategies.

[View the original repository on GitHub](https://github.com/ZZZSimulator/ZSim)

## Key Features:

*   **Automated Battle Simulation:**  Automatically simulates battles based on Action Priority Lists (APLs).
*   **Damage Calculation:** Accurately calculates total damage output for your team compositions.
*   **Detailed Reporting:** Generates visual charts and tables with in-depth damage information per character.
*   **Equipment Editing:**  Easily edit agent equipment within the simulator.
*   **APL Customization:** Edit Action Priority List (APL) code to refine your strategies.
*   **User-Friendly Interface:**  Provides an intuitive interface for easy use.

## Installation

### Prerequisites: Install `uv` (Universal Virtualenv)

Choose one of the following installation methods for `uv`:

```bash
# Using pip if you have Python installed:
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
# On lower versions of Windows:
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

For more detailed instructions, refer to the official `uv` installation guide: [https://docs.astral.sh/uv/getting-started/installation/](https://docs.astral.sh/uv/getting-started/installation/)

### Install and Run ZSim

1.  **Navigate:** Open your terminal and change the directory to the ZSim project folder.
2.  **Sync Dependencies:**
    ```bash
    uv sync
    ```
3.  **Run the Simulator:**
    ```bash
    uv run zsim run
    ```

## Development

### Key Components

*   **Simulation Engine:** Core battle simulation logic in `zsim/simulator/`.
*   **Web API:** FastAPI-based REST API in `zsim/api_src/`.
*   **Web UI:** Streamlit-based interface (`zsim/webui.py`) and a new Vue.js + Electron desktop application (`electron-app/`).
*   **CLI:** Command-line interface via `zsim/run.py`.
*   **Database:** SQLite-based storage for character and enemy configurations.
*   **Electron App:** Desktop application built with Vue.js and Electron, communicating with the FastAPI backend.

### Setup and Installation (Development)

```bash
# Install UV package manager first (if not already installed)
uv sync
# For WebUI development:
uv run zsim run
# For FastAPI backend:
uv run zsim api

# For Electron App development, install Node.js dependencies:
cd electron-app
corepack install
pnpm install
```

### Testing Structure

*   **Unit Tests:** Located in the `tests/` directory.
*   **API Tests:** Located in the `tests/api/` directory.
*   **Fixtures:** Defined in `tests/conftest.py`.
*   **Testing Framework:** Uses `pytest` with `asyncio` support.

### Running Tests

```bash
# Run the tests
uv run pytest
# Run the tests with coverage report
uv run pytest -v --cov=zsim --cov-report=html
```

## TODO

For more detailed information and contribution guidelines, please consult the [Develop Guide](https://github.com/ZZZSimulator/ZSim/wiki/%E8%B4%A1%E7%8C%AE%E6%8C%87%E5%8D%97-Develop-Guide).