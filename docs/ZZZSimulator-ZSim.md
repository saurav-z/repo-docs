# ZSim: Unleash Your Zenless Zone Zero Team's Full Potential

**Maximize your team's damage output in Zenless Zone Zero with ZSim, a powerful and user-friendly battle simulator!** [View the project on GitHub](https://github.com/ZZZSimulator/ZSim)

![ZSim Logo](./docs/img/横板logo成图.png)

## What is ZSim?

ZSim is a comprehensive battle simulator and damage calculator specifically designed for Zenless Zone Zero (ZZZ), the action game from Hoyoverse. It allows you to analyze your team compositions, optimize gear, and understand damage output with ease. ZSim automatically simulates battles based on your chosen Action Priority List (APL), removing the need for manual skill sequence input.

## Key Features

*   **Automated Battle Simulation:**  Simulates battles automatically based on APLs, eliminating manual input.
*   **Team Damage Calculation:** Accurately calculates total damage output for your selected team.
*   **Visual Reporting:** Generates intuitive charts and tables for easy result analysis.
*   **Detailed Damage Breakdown:** Provides in-depth damage information for each character.
*   **Agent Customization:** Allows you to edit and customize your agents' equipment.
*   **APL Customization:** Edit APL code to fine-tune your team's actions.

## Installation

### Prerequisites: Install UV (if you haven't already)

UV is a fast Python package and virtual environment manager.  Choose your operating system below to install UV:

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
# On lower version of Windows:
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

For detailed instructions, see the official UV installation guide: [https://docs.astral.sh/uv/getting-started/installation/](https://docs.astral.sh/uv/getting-started/installation/)

### Install and Run ZSim

1.  **Get the Source Code:** Download the latest source code from the release page or use `git clone`.
2.  **Navigate to Project Directory:** Open a terminal in the project's directory.
3.  **Install Dependencies and Run:**

    ```bash
    uv sync
    uv run zsim run
    ```

## Development

### Key Components

*   **Simulation Engine:** Core battle logic resides in `zsim/simulator/`.
*   **Web API:**  REST API built with FastAPI (`zsim/api_src/`).
*   **Web UI:**  Streamlit-based interface (`zsim/webui.py`) and new Vue.js + Electron desktop application (`electron-app/`).
*   **CLI:** Command-line interface via `zsim/run.py`.
*   **Database:** SQLite for character/enemy configurations.
*   **Electron App:** Desktop application (Vue.js & Electron) communicating with the FastAPI backend.

### Setup and Installation for Development

```bash
# Install UV package manager first
uv sync
# For WebUI development
uv run zsim run
# For FastAPI backend
uv run zsim api

# For Electron App development, also install Node.js dependencies
cd electron-app
yarn install
```

### Testing Structure

*   Unit tests in `tests/` directory.
*   API tests in `tests/api/`.
*   Fixtures defined in `tests/conftest.py`.
*   Uses pytest with asyncio support.

```bash
# Run the tests
uv run pytest
# Run the tests with coverage report
uv run pytest -v --cov=zsim --cov-report=html
```

## Contribute

See our [Development Guide](https://github.com/ZZZSimulator/ZSim/wiki/%E8%B4%A1%E7%8C%AE%E6%8C%87%E5%8D%97-Develop-Guide) for more details on how you can contribute!