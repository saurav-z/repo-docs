# ZSim: The Ultimate Zenless Zone Zero Damage Calculator and Battle Simulator

**Unleash the power of your Zenless Zone Zero teams with ZSim, the automatic battle simulator and damage calculator, providing in-depth analysis for optimized gameplay.**  [Explore the original repository](https://github.com/ZZZSimulator/ZSim)

## Key Features:

*   **Automatic Battle Simulation:**  Eliminates the need for manual skill sequence input, streamlining your analysis (sequence mode available upon request).
*   **Comprehensive Damage Calculation:** Accurately calculates total damage output, considering character weapons, equipment, and team composition.
*   **Visualized Results:** Generates easy-to-understand charts and tables for in-depth damage information.
*   **Agent Customization:** Allows detailed editing of agent equipment and Action Priority Lists (APLs).
*   **User-Friendly Interface:** Offers a streamlined interface for effortless team optimization.

## Installation Guide

### Prerequisites: UV Package Manager

ZSim utilizes the `uv` package manager for dependency management.  If you haven't already, install it using one of the following methods:

```bash
# Using pip if you have python installed:
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

For a comprehensive installation guide, please consult the official UV documentation: [https://docs.astral.sh/uv/getting-started/installation/](https://docs.astral.sh/uv/getting-started/installation/)

### Installing and Running ZSim

1.  **Get the Source Code:** Download the latest source code from the [releases page](https://github.com/ZZZSimulator/ZSim/releases) or use `git clone`.
2.  **Navigate to the Project Directory:** Open your terminal and navigate to the directory where you downloaded the ZSim project.
3.  **Install Dependencies and Run:** Execute the following commands in your terminal:

    ```bash
    uv sync
    uv run zsim run
    ```

## Development Information

### Core Components:

*   **Simulation Engine:** Located in `zsim/simulator/`.
*   **Web API:** Built with FastAPI, found in `zsim/api_src/`.
*   **Web UI:**  Uses Streamlit (`zsim/webui.py`) and a new Vue.js + Electron desktop application in `electron-app/`.
*   **CLI:** Command-line interface via `zsim/run.py`.
*   **Database:**  Uses SQLite for storing character and enemy configurations.
*   **Electron App:**  A desktop application built with Vue.js and Electron that communicates with the FastAPI backend.

### Development Setup:

1.  **Install UV package manager first (see installation guide above).**
2.  **For WebUI development:**

    ```bash
    uv sync
    uv run zsim run
    ```
3.  **For FastAPI backend development:**

    ```bash
    uv sync
    uv run zsim api
    ```
4.  **For Electron App development, also install Node.js dependencies:**

    ```bash
    cd electron-app
    corepack install
    pnpm install
    ```

### Testing:

*   **Unit Tests:** Located in the `tests/` directory.
*   **API Tests:** Found in `tests/api/`.
*   **Fixtures:** Defined in `tests/conftest.py`.
*   Uses pytest with asyncio support

#### Running Tests:

```bash
# Run the tests
uv run pytest
# Run the tests with coverage report
uv run pytest -v --cov=zsim --cov-report=html
```

## TODO List:

For more details on future development plans, please refer to the [Development Guide](https://github.com/ZZZSimulator/ZSim/wiki/%E8%B4%A1%E7%8C%AE%E6%8C%87%E5%8D%97-Develop-Guide).