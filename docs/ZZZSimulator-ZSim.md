# ZSim: Zenless Zone Zero Battle Simulator & Damage Calculator

**Maximize your team's potential in Zenless Zone Zero with ZSim, the powerful and automated battle simulator.**  [View the original repository](https://github.com/ZZZSimulator/ZSim)

![ZSim Logo](docs/img/横板logo成图.png)

## About ZSim

ZSim is a comprehensive battle simulator and damage calculator specifically designed for Zenless Zone Zero (ZZZ), the action role-playing game from Hoyoverse. This tool allows you to analyze your team compositions and optimize your strategies without manual skill sequence input (sequence mode can be added if desired). Simply equip your agents, select an Action Priority List (APL), and let ZSim automatically simulate battles, generate detailed reports, and visualize your team's performance.

## Key Features

*   **Automated Battle Simulation:**  Focus on strategy, not tedious setup.
*   **Total Damage Calculation:** Accurately assess your team's overall damage output.
*   **Visual Charts & Reports:**  Gain clear insights through intuitive data visualizations.
*   **Detailed Damage Breakdown:** Analyze each character's contribution.
*   **Agent Equipment Editor:** Customize your characters' gear for optimal performance.
*   **APL Customization:**  Fine-tune your team's actions with customizable APLs.

## Installation

### Prerequisites: Install UV Package Manager

ZSim utilizes the `uv` package manager for dependency management. If you haven't already, install it using one of the following methods:

```bash
# Using pip (if you have Python installed):
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
# On older Windows versions:
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

For more detailed installation instructions, please refer to the official `uv` documentation: [https://docs.astral.sh/uv/getting-started/installation/](https://docs.astral.sh/uv/getting-started/installation/)

### Installing and Running ZSim

1.  **Get the Code:** Download the latest release from the releases page or clone the repository using `git clone`.
2.  **Navigate to the Project Directory:** Open a terminal and navigate to the directory where you downloaded ZSim.
3.  **Install Dependencies & Run:**
    ```bash
    uv sync
    uv run zsim run
    ```

## Development

### Core Components

*   **Simulation Engine:** Located in `zsim/simulator/`, handles the core battle logic.
*   **Web API:** Built with FastAPI, providing a REST API at `zsim/api_src/`.
*   **Web UI:** Uses Streamlit ( `zsim/webui.py` ) and a new Vue.js + Electron desktop application in `electron-app/` for the user interface.
*   **CLI:** Command-line interface available through `zsim/run.py`.
*   **Database:** Utilizes SQLite for storing character and enemy configurations.
*   **Electron App:** A desktop application built with Vue.js and Electron, interacting with the FastAPI backend.

### Setup and Development

To get started with development:

```bash
# Install dependencies
uv sync

# Run the WebUI
uv run zsim run

# Run the FastAPI backend
uv run zsim api

# For Electron App development, also install Node.js dependencies
cd electron-app
yarn install
```

### Testing

ZSim uses pytest for testing, with unit tests, API tests, and fixtures.

```bash
# Run tests
uv run pytest

# Run tests with coverage report
uv run pytest -v --cov=zsim --cov-report=html
```

## Further Information

For detailed information on contributing and development, please consult the [develop guide](https://github.com/ZZZSimulator/ZSim/wiki/%E8%B4%A1%E7%8C%AE%E6%8C%87%E5%8D%97-Develop-Guide) within the project's wiki.