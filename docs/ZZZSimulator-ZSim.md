# ZSim: Zenless Zone Zero Battle Simulator & Damage Calculator

Unleash the power of your Zenless Zone Zero teams with ZSim, a comprehensive battle simulator and damage calculator! ([View on GitHub](https://github.com/ZZZSimulator/ZSim))

## Key Features

*   **Automated Battle Simulation:** Automatically simulates battles based on your team composition and equipment, eliminating the need for manual skill sequence input.
*   **Detailed Damage Calculation:** Calculates total damage output, considering character stats, weapon effects, and equipment.
*   **Visual Reporting:** Generates visual charts and tables to provide insights into your team's performance.
*   **Agent Equipment Customization:** Edit and optimize your agents' equipment to maximize damage potential.
*   **APL (Action Priority List) Editing:** Customize the APL to fine-tune your team's actions and strategies.

## Installation

### Prerequisites

Before installing ZSim, ensure you have the `uv` package manager installed.  Follow the instructions for your operating system below:

**Install UV (if you haven't already)**

**Using pip if you have python installed:**

```bash
pip install uv
```

**On macOS or Linux:**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**On Windows11 24H2 or later:**

```bash
winget install --id=astral-sh.uv  -e
```

**On lower version of Windows:**

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Or check the official installation guide: [https://docs.astral.sh/uv/getting-started/installation/](https://docs.astral.sh/uv/getting-started/installation/)

### Install and Run ZSim

1.  **Download:** Obtain the latest source code from the [releases page](https://github.com/ZZZSimulator/ZSim/releases) or use `git clone`.
2.  **Navigate:** Open a terminal in the project's directory.
3.  **Install Dependencies & Run:**

```bash
uv sync
uv run zsim run
```

## Development

### Key Components

*   **Simulation Engine:** Core battle logic in `zsim/simulator/`.
*   **Web API:** FastAPI-based REST API in `zsim/api_src/`.
*   **Web UI:** Streamlit-based interface in `zsim/webui.py` and new Vue.js + Electron desktop application in `electron-app/`.
*   **CLI:** Command-line interface via `zsim/run.py`.
*   **Database:** SQLite-based storage for character/enemy configurations.
*   **Electron App:** Desktop application built with Vue.js and Electron, communicating with the FastAPI backend.

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
*   **Testing Framework:** Uses `pytest` with `asyncio` support.

### Running Tests

```bash
# Run the tests
uv run pytest
# Run the tests with coverage report
uv run pytest -v --cov=zsim --cov-report=html
```

## Further Information

Check the [develop guide](https://github.com/ZZZSimulator/ZSim/wiki/%E8%B4%A1%E7%8C%AE%E6%8C%87%E5%8D%97-Develop-Guide) for details on contributing and the project's roadmap.