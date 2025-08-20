# ZSim: Zenless Zone Zero Battle Simulator & Damage Calculator

**Maximize your team's potential in Zenless Zone Zero with ZSim, the automated battle simulator and damage calculator!** ([Original Repository](https://github.com/ZZZSimulator/ZSim))

<img src="./docs/img/横板logo成图.png" alt="ZSim Logo" width="400"/>

## Key Features:

*   **Automated Battle Simulation:** No manual skill sequence input needed (though sequence mode is planned for future iterations).
*   **Team Damage Calculation:** Determine total damage output based on your team composition and equipment.
*   **Visual Reporting:** Generate clear charts and tables for detailed damage analysis.
*   **Agent Customization:** Edit agent equipment to optimize performance.
*   **Action Priority List (APL) Customization:**  Modify APL code for advanced strategy.
*   **Detailed Damage Breakdown:** Analyze damage information for each character.

## Installation

ZSim is easy to set up. Follow these steps:

### 1. Install UV (if you haven't already)

UV is a fast Python package installer and resolver.

Open your terminal and use one of the following commands:

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

For more detailed instructions, consult the official UV documentation:  [https://docs.astral.sh/uv/getting-started/installation/](https://docs.astral.sh/uv/getting-started/installation/)

### 2. Install and Run ZSim

1.  **Download:** Get the latest source code from the release page or use `git clone`.
2.  **Navigate:** Open your terminal in the project directory.
3.  **Install Dependencies and Run:**

```bash
uv sync
uv run zsim run
```

## Development

ZSim's architecture comprises several key components:

*   **Simulation Engine:**  Core battle simulation logic (`zsim/simulator/`).
*   **Web API:** FastAPI-based REST API for programmatic access (`zsim/api_src/`).
*   **Web UI:** Streamlit interface (`zsim/webui.py`) and a new Vue.js + Electron desktop application (`electron-app/`).
*   **CLI:** Command-line interface (`zsim/run.py`).
*   **Database:** SQLite-based storage for character and enemy data.
*   **Electron App:** Desktop application using Vue.js and Electron, communicating with the FastAPI backend.

### Setup and Installation (Development)

```bash
# Install UV package manager first
uv sync
# For WebUI develop
uv run zsim run 
# For FastAPI backend
uv run zsim api

# For Electron App development, also install Node.js dependencies
cd electron-app
yarn install
```

### Testing

ZSim utilizes pytest for testing:

*   Unit tests in `tests/`
*   API tests in `tests/api/`
*   Fixtures defined in `tests/conftest.py`
*   pytest with asyncio support

```bash
# Run the tests
uv run pytest
# Run the tests with coverage report
uv run pytest -v --cov=zsim --cov-report=html
```

## Future Development

See the [develop guide](https://github.com/ZZZSimulator/ZSim/wiki/%E8%B4%A1%E7%8C%AE%E6%8C%87%E5%8D%97-Develop-Guide) for more information.