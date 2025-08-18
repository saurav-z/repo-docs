# ZSim: Zenless Zone Zero Battle Simulator & Damage Calculator

**Unleash your team's potential and dominate the battlefield in Zenless Zone Zero with ZSim, a powerful and automated battle simulator!**  ([Original Repository](https://github.com/ZZZSimulator/ZSim))

## About ZSim

ZSim is an automated battle simulator and damage calculator specifically designed for the action-packed game Zenless Zone Zero (ZZZ). It simplifies the complex process of optimizing your team's damage output by automatically simulating battles based on pre-defined Action Priority Lists (APLs).  ZSim empowers you to experiment with different character builds and team compositions without tedious manual input, providing in-depth analysis and visual reports.

## Key Features

*   **Automated Battle Simulation:**  No need to manually input skill sequences; ZSim handles the action automatically.
*   **Comprehensive Damage Calculation:** Accurately calculates total damage output, considering character weapons, equipment, and team synergies.
*   **User-Friendly Interface:**  Easily edit agent equipment, select APLs, and run simulations.
*   **Visual Reporting:**  Generates insightful charts and tables for detailed damage analysis.
*   **Character-Specific Damage Breakdown:**  Provides granular damage information for each character in your team.
*   **APL Customization:** Edit or create your own APL codes to fine-tune your strategy.

## Installation

### Prerequisites

Before installing ZSim, you need to install the `uv` package manager. Follow the instructions below based on your operating system:

**Using pip (if you have python installed):**

```bash
pip install uv
```

**On macOS or Linux:**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**On Windows 11 24H2 or later:**

```bash
winget install --id=astral-sh.uv  -e
```

**On lower version of Windows:**

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Installing and Running ZSim

1.  **Download:**  Download the latest source code from the [release page](https://github.com/ZZZSimulator/ZSim/releases) or clone the repository using `git clone`.

2.  **Navigate:** Open a terminal in the project directory.

3.  **Install Dependencies:**

    ```bash
    uv sync
    ```

4.  **Run the Simulator:**

    ```bash
    uv run zsim run
    ```

## Development

### Key Components

ZSim's functionality is divided into several key components:

1.  **Simulation Engine:**  Core logic in `zsim/simulator/` for battle simulation.
2.  **Web API:**  FastAPI-based REST API in `zsim/api_src/` for programmatic access.
3.  **Web UI:**  Streamlit-based interface in `zsim/webui.py` and a new Vue.js + Electron desktop application in `electron-app/`.
4.  **CLI:** Command-line interface via `zsim/run.py`.
5.  **Database:** SQLite-based storage for character and enemy configurations.
6.  **Electron App:** Desktop application built with Vue.js and Electron that communicates with the FastAPI backend.

### Setup and Installation for Development

```bash
# Install UV package manager first (if not already done)
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

*   Unit tests are located in the `tests/` directory.
*   API tests are in `tests/api/`.
*   Fixtures are defined in `tests/conftest.py`.
*   Testing uses pytest with asyncio support.

```bash
# Run all tests
uv run pytest

# Run tests with coverage report
uv run pytest -v --cov=zsim --cov-report=html
```

## Further Development

For more information and contributions, check the [develop guide](https://github.com/ZZZSimulator/ZSim/wiki/%E8%B4%A1%E7%8C%AE%E6%8C%87%E5%8D%97-Develop-Guide).