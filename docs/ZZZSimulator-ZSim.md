# ZSim: The Ultimate Zenless Zone Zero Damage Calculator & Battle Simulator

**Maximize your team's performance in Zenless Zone Zero with ZSim, a powerful, automated damage calculator and battle simulator.** ([Original Repository](https://github.com/ZZZSimulator/ZSim))

## Key Features:

*   **Automated Battle Simulation:** Automatically simulates battles based on your team composition and equipment, eliminating the need for manual skill sequence input.
*   **Comprehensive Damage Calculation:** Accurately calculates total damage output, considering character weapons, equipment, and team buffs.
*   **User-Friendly Interface:** Features an intuitive interface, including a Streamlit-based web UI and a new Electron desktop application.
*   **Visualized Results:** Generates easy-to-understand visual charts and detailed damage reports.
*   **Customizable Agent Configuration:** Allows you to edit agent equipment and APL (Action Priority List) codes for in-depth analysis.

## Installation

### Prerequisites: Install UV (Universal Virtual Environment)

You'll need the UV package manager to install ZSim.  Choose the installation method for your operating system:

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

For further information, see the official UV installation guide: [https://docs.astral.sh/uv/getting-started/installation/](https://docs.astral.sh/uv/getting-started/installation/)

### Install and Run ZSim

1.  **Download the Source Code:** Obtain the latest release from the [release page](link to releases, if available) or clone the repository using `git clone`.
2.  **Navigate to Project Directory:** Open a terminal in the ZSim project directory.
3.  **Run ZSim:**

```bash
uv sync
uv run zsim run
```

## Development

### Key Components

*   **Simulation Engine:**  `zsim/simulator/` - Core battle simulation logic.
*   **Web API:** `zsim/api_src/` - FastAPI-based REST API.
*   **Web UI:** `zsim/webui.py` - Streamlit-based interface, and the new Vue.js + Electron desktop application in `electron-app/`.
*   **CLI:** `zsim/run.py` - Command-line interface.
*   **Database:** SQLite database for character and enemy data.
*   **Electron App:** Desktop application built with Vue.js and Electron, interacting with the FastAPI backend.

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
corepack install
pnpm install
```

### Testing Structure

*   **Unit Tests:** Located in the `tests/` directory.
*   **API Tests:** Located in `tests/api/`.
*   **Fixtures:** Defined in `tests/conftest.py`.
*   **Test Runner:** Uses pytest with asyncio support.

```bash
# Run the tests
uv run pytest
# Run the tests with coverage report
uv run pytest -v --cov=zsim --cov-report=html
```

##  Contributing

For details on contributing, please refer to the [develop guide](https://github.com/ZZZSimulator/ZSim/wiki/%E8%B4%A1%E7%8C%AE%E6%8C%87%E5%8D%97-Develop-Guide).