# ZSim: Zenless Zone Zero Battle Simulator & Damage Calculator

**Maximize your team's potential in Zenless Zone Zero with ZSim, the automated battle simulator and damage calculator.** ([View the original repository](https://github.com/ZZZSimulator/ZSim))

![ZSim Project Logo](docs/img/横板logo成图.png)

## Introduction

ZSim is a powerful tool designed for Zenless Zone Zero (ZZZ), an action role-playing game by Hoyoverse. This simulator automatically calculates damage output, taking into account character equipment and team composition. Based on a predefined Action Priority List (APL), ZSim simulates battles, analyzes results, and provides clear visual reports.

## Key Features

*   **Automated Damage Calculation:** Effortlessly calculates total team damage output.
*   **Visual Reporting:** Generates detailed charts and tables for easy analysis.
*   **Character Customization:** Allows editing of agent equipment for precise simulations.
*   **APL Customization:** Enables modification of Action Priority Lists for advanced strategies.
*   **User-Friendly Interface:** Provides an intuitive interface for easy use.
*   **Detailed Damage Breakdown:** Offers comprehensive damage information per character.

## Installation

### Prerequisites

Before installing ZSim, ensure you have the UV package manager installed.  Follow the instructions below based on your operating system.

**Install UV:**

*   **Using pip (if you have Python installed):**
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

    For more details on UV installation, please check the official guide: [https://docs.astral.sh/uv/getting-started/installation/](https://docs.astral.sh/uv/getting-started/installation/)

### Install and Run ZSim

1.  **Navigate to the Project Directory:** Open your terminal and change the directory to where you cloned the ZSim repository.
2.  **Install Dependencies:**
    ```bash
    uv sync
    ```
3.  **Run the Simulator:**
    ```bash
    uv run zsim run
    ```

## Development

### Key Components

*   **Simulation Engine:** Core logic within `zsim/simulator/` manages battle simulations.
*   **Web API:** A FastAPI-based REST API resides in `zsim/api_src/`, enabling programmatic access.
*   **Web UI:** Interfaces are built using Streamlit (`zsim/webui.py`) and a new Vue.js + Electron desktop application (`electron-app/`).
*   **CLI:** Command-line interface via `zsim/run.py`.
*   **Database:** SQLite is utilized for storing character and enemy configurations.
*   **Electron App:** A desktop application, built with Vue.js and Electron, that communicates with the FastAPI backend.

### Development Setup

```bash
# Install UV package manager (if not already installed)
uv sync

# For WebUI development
uv run zsim run

# For FastAPI backend
uv run zsim api

# For Electron App development, install Node.js dependencies
cd electron-app
corepack install
pnpm install
```

### Testing

*   **Unit Tests:** Located in the `tests/` directory.
*   **API Tests:** Found in `tests/api/`.
*   **Fixtures:** Defined in `tests/conftest.py`.
*   **Testing Framework:** Uses pytest with asyncio support.

```bash
# Run the tests
uv run pytest

# Run the tests with coverage report
uv run pytest -v --cov=zsim --cov-report=html
```

## TODO List

Refer to the [Develop Guide](https://github.com/ZZZSimulator/ZSim/wiki/%E8%B4%A1%E7%8C%AE%E6%8C%87%E5%8D%97-Develop-Guide) for more information.