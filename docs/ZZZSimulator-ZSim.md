# ZSim: Zenless Zone Zero Battle Simulator & Damage Calculator

**Unleash your team's full potential with ZSim, a powerful battle simulator and damage calculator for Zenless Zone Zero, helping you optimize your team compositions.** ([View the original repo](https://github.com/ZZZSimulator/ZSim))

![ZSim Logo](docs/img/横板logo成图.png)

## About ZSim

ZSim is a sophisticated battle simulator and damage calculator designed specifically for the fast-paced action of Hoyoverse's Zenless Zone Zero (ZZZ). ZSim allows you to analyze your team's total damage output by taking into account each agent's weapon and equipment characteristics. Using preset Action Priority Lists (APLs), ZSim **automatically simulates** the battle, triggers buffs, and records your results, generating reports with visual charts and comprehensive tables.

## Key Features

*   **Automatic Simulation:** No need to manually configure skill sequences (though sequence mode is planned!).
*   **Team Damage Calculation:** Accurately calculates total damage output based on team composition.
*   **Visual Reports:** Generates insightful charts and tables for easy analysis.
*   **Detailed Damage Breakdown:** Provides in-depth damage information for each character in your team.
*   **Agent Equipment Customization:** Allows you to edit and optimize your agents' equipment.
*   **APL Customization:** Edit the Action Priority List code to finely tune your team's strategy.

## Installation

### Prerequisites: UV Package Manager

ZSim utilizes the UV package manager for dependency management. Ensure UV is installed on your system before proceeding.

**Install UV:**

*   **Using pip (if Python is installed):**
    ```bash
    pip install uv
    ```
*   **macOS or Linux:**
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```
*   **Windows 11 24H2 or later:**
    ```bash
    winget install --id=astral-sh.uv  -e
    ```
*   **Older Windows versions:**
    ```powershell
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    ```
*   **For detailed instructions, visit the official UV documentation:** [https://docs.astral.sh/uv/getting-started/installation/](https://docs.astral.sh/uv/getting-started/installation/)

### Installing and Running ZSim

1.  **Download the Latest Code:** Download the latest source code from the releases page or use `git clone`.
2.  **Navigate to Project Directory:** Open your terminal and navigate to the project's root directory.
3.  **Install Dependencies and Run:**

    ```bash
    uv sync
    uv run zsim run
    ```

## Development Information

### Key Components

*   **Simulation Engine:** Core battle simulation logic within the `zsim/simulator/` directory.
*   **Web API:** A REST API built using FastAPI, found in `zsim/api_src/`, for programmatic access.
*   **Web UI:** A user-friendly interface built with Streamlit (`zsim/webui.py`) and a new Vue.js + Electron desktop application (`electron-app/`).
*   **CLI:** Command-line interface through `zsim/run.py`.
*   **Database:** SQLite is used for character and enemy configuration storage.
*   **Electron App:** Desktop application built with Vue.js and Electron that communicates with the FastAPI backend

### Setup and Installation for Development

1.  **Install UV (if you haven't already - see Installation section):** `uv sync`
2.  **For WebUI Development:** `uv run zsim run`
3.  **For FastAPI Backend Development:** `uv run zsim api`
4.  **For Electron App Development:**
    ```bash
    cd electron-app
    yarn install
    ```

### Testing

*   **Unit Tests:** Located in the `tests/` directory.
*   **API Tests:** Found in `tests/api/`.
*   **Fixtures:** Defined in `tests/conftest.py`.
*   **Test Framework:** Uses pytest with asyncio support.

**Running Tests:**

*   **Run All Tests:** `uv run pytest`
*   **Run Tests with Coverage Report:** `uv run pytest -v --cov=zsim --cov-report=html`

## TODO List

Refer to the [development guide](https://github.com/ZZZSimulator/ZSim/wiki/%E8%B4%A1%E7%8C%AE%E6%8C%87%E5%8D%97-Develop-Guide) for more details on future development plans.