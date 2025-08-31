# ZSim: Zenless Zone Zero Damage Calculator & Battle Simulator

**Maximize your Zenless Zone Zero team's potential with ZSim, the automated battle simulator designed to optimize your damage output.** [Visit the original repository](https://github.com/ZZZSimulator/ZSim)

![ZSim Logo](docs/img/横板logo成图.png)

## Introduction

ZSim is a comprehensive battle simulator and damage calculator specifically designed for Zenless Zone Zero (ZZZ), the action game from Hoyoverse.  It automates the simulation process, allowing you to focus on building the perfect team.

## Key Features:

*   **Automated Battle Simulation:**  No manual skill sequence setup is required.
*   **Team Damage Calculation:** Accurately calculates total damage based on team composition.
*   **Visual Damage Reports:** Generates easy-to-understand charts and tables for detailed analysis.
*   **Agent Equipment Editing:** Allows you to customize your agents' equipment for optimal performance.
*   **Action Priority List (APL) Customization:**  Edit APL code to fine-tune your team's actions.
*   **Detailed Character Damage Information:** Provides in-depth damage breakdowns for each character.
*   **User-Friendly Interface:** Simple and intuitive interface for easy operation.

## Installation

### Prerequisites:

Before installing ZSim, ensure you have the following installed:

*   **Python:** Make sure you have Python installed on your system.
*   **UV Package Manager:**  Install the UV package manager using one of the following methods.

    *   **Using pip (if you have Python installed):**
        ```bash
        pip install uv
        ```
    *   **On macOS or Linux:**
        ```bash
        curl -LsSf https://astral.sh/uv/install.sh | sh
        ```
    *   **On Windows 11 24H2 or later:**
        ```bash
        winget install --id=astral-sh.uv  -e
        ```
    *   **On lower versions of Windows:**
        ```powershell
        powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
        ```
    *   For more detailed installation instructions, refer to the official UV documentation: [https://docs.astral.sh/uv/getting-started/installation/](https://docs.astral.sh/uv/getting-started/installation/)

### ZSim Installation & Run

1.  **Download:** Download the latest source code from the [release page](https://github.com/ZZZSimulator/ZSim/releases) or clone the repository using Git:

    ```bash
    git clone https://github.com/ZZZSimulator/ZSim.git
    cd ZSim
    ```

2.  **Install dependencies and Run:** Open your terminal in the project directory and run the following commands:

    ```bash
    uv sync
    uv run zsim run
    ```

## Development

### Key Components

*   **Simulation Engine:**  `zsim/simulator/` - Core logic for battle simulation.
*   **Web API:** `zsim/api_src/` - FastAPI-based REST API for programmatic access.
*   **Web UI:** `zsim/webui.py` (Streamlit) and  `electron-app/` (Vue.js + Electron desktop application) - User interfaces.
*   **CLI:** `zsim/run.py` - Command-line interface.
*   **Database:** SQLite for character/enemy configurations.
*   **Electron App:** Desktop application built with Vue.js and Electron that communicates with the FastAPI backend

### Setup and Installation

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

### Testing Structure

*   Unit tests in `tests/` directory
*   API tests in `tests/api/`
*   Fixtures defined in `tests/conftest.py`
*   Uses pytest with asyncio support

```bash
# Run the tests
uv run pytest
# Run the tests with coverage report
uv run pytest -v --cov=zsim --cov-report=html
```

## Further Information

For details, see [develop guide](https://github.com/ZZZSimulator/ZSim/wiki/%E8%B4%A1%E7%8C%AE%E6%8C%87%E5%8D%97-Develop-Guide) for more details.