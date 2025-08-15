# ZSim: Zenless Zone Zero Battle Simulator & Damage Calculator

**Maximize your team's potential in Zenless Zone Zero with ZSim, the automated battle simulator and damage calculator!**

[View the Original Repo on GitHub](https://github.com/ZZZSimulator/ZSim)

## About ZSim

ZSim is a powerful tool designed to analyze and optimize your team compositions in Zenless Zone Zero (ZZZ), the action game from Hoyoverse.  This simulator automatically calculates total damage output, considering character equipment and action priority lists (APLs), without requiring manual skill sequence input.  ZSim provides a user-friendly interface to experiment with different builds and strategies, giving you a competitive edge in the game.

## Key Features

*   **Automated Damage Calculation:** Automatically simulates battles based on preset Action Priority Lists (APLs).
*   **Team Composition Analysis:** Calculate total damage output for your chosen team.
*   **Visual Reporting:** Generates comprehensive reports with visual charts and tables for easy analysis.
*   **Detailed Damage Information:** Provides in-depth damage breakdowns for each character.
*   **Equipment Customization:**  Edit agent equipment to experiment with various builds.
*   **APL Editing:**  Customize APL codes to fine-tune your team's actions.

## Installation

### Prerequisites

Before installing ZSim, ensure you have the following installed:

*   **UV Package Manager:**  ZSim utilizes the `uv` package manager for dependency management. Install it using one of the following methods:

    *   **Using `pip` (if Python is installed):**
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

    *   **Older Windows Versions:**
        ```powershell
        powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
        ```

    *   For detailed instructions on installing `uv`, please refer to the official documentation: [https://docs.astral.sh/uv/getting-started/installation/](https://docs.astral.sh/uv/getting-started/installation/)

### Install and Run ZSim

1.  **Clone the Repository or Download Source Code:** Get the latest version of the ZSim source code from the release page or by using `git clone`.
2.  **Navigate to the Project Directory:** Open a terminal and navigate to the directory where you've saved the ZSim project.
3.  **Install Dependencies and Run the Simulator:**

    ```bash
    uv sync
    uv run zsim run
    ```

## Development

### Core Components

*   **Simulation Engine:** Core battle simulation logic located in `zsim/simulator/`.
*   **Web API:** FastAPI-based REST API in `zsim/api_src/` for programmatic access.
*   **Web UI:**  Streamlit-based user interface in `zsim/webui.py`, and a new Vue.js + Electron desktop application in `electron-app/`.
*   **CLI:** Command-line interface accessible via `zsim/run.py`.
*   **Database:** SQLite database for storing character and enemy configurations.
*   **Electron App:** Desktop application developed with Vue.js and Electron, communicating with the FastAPI backend.

### Setup for Development

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

### Testing

*   **Unit Tests:** Located in the `tests/` directory.
*   **API Tests:** Located in the `tests/api/` directory.
*   **Fixtures:** Defined in `tests/conftest.py`.
*   **Testing Framework:** Uses pytest with asyncio support.

```bash
# Run Tests
uv run pytest
# Run Tests with Coverage Report
uv run pytest -v --cov=zsim --cov-report=html
```

## TODO List

For more details on development and future features, please consult the [Develop Guide](https://github.com/ZZZSimulator/ZSim/wiki/%E8%B4%A1%E7%8C%AE%E6%8C%87%E5%8D%97-Develop-Guide).