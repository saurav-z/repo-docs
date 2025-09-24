# ZSim: Your Ultimate Zenless Zone Zero Battle Simulator & Damage Calculator

Unleash the power of strategic team building with ZSim, the comprehensive battle simulator and damage calculator for Hoyoverse's Zenless Zone Zero.  ([View the original repo](https://github.com/ZZZSimulator/ZSim))

## Key Features

*   **Automated Battle Simulation:** Automatically simulates battles based on your team's action priority list (APL), eliminating the need for manual skill sequence input.
*   **Detailed Damage Calculation:**  Calculates total damage output, considering character stats, weapons, and equipment.
*   **Visual Data & Reporting:** Generates clear visual charts and tables to analyze your team's performance.
*   **Agent Customization:**  Allows you to easily edit and optimize your agents' equipment.
*   **APL Customization:**  Provides the flexibility to modify APL code for tailored simulations.
*   **User-Friendly Interface:** Offers an intuitive interface, making it easy to calculate and analyze team compositions.

## Getting Started

### Installation

1.  **Install UV Package Manager (if you haven't already):**
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
    For more detailed installation instructions for UV, please visit: [https://docs.astral.sh/uv/getting-started/installation/](https://docs.astral.sh/uv/getting-started/installation/)

2.  **Install and Run ZSim:**
    ```bash
    # Navigate to your project directory in terminal
    uv sync
    uv run zsim run
    ```

## Development

### Components

ZSim is built with a modular architecture, comprising several key components:

*   **Simulation Engine:** Core battle logic in `zsim/simulator/`.
*   **Web API:** FastAPI-based REST API in `zsim/api_src/`.
*   **Web UI:** Streamlit interface (`zsim/webui.py`) and a new Vue.js + Electron desktop application (`electron-app/`).
*   **CLI:** Command-line interface via `zsim/run.py`.
*   **Database:** SQLite-based storage for character and enemy configurations.
*   **Electron App:** Desktop application built with Vue.js and Electron, interacting with the FastAPI backend.

### Build System

A comprehensive Make-based build system is employed for development, building, and release processes.

#### Make Targets

*   `make build`: Full build (clean + backend + electron).
*   `make backend`: Builds the backend API only.
*   `make electron-build`: Builds the Electron desktop application only.
*   `make dev`: Starts the frontend development server.
*   `make clean`: Cleans all build files.
*   `make check`: Checks dependencies.
*   `make help`: Displays help information.

### Setup & Installation (Development)
```bash
# Install UV package manager first
uv sync

# For WebUI develop
uv run zsim run 
# For FastAPI backend
uv run zsim api

# For Electron App development, also install Node.js dependencies
cd electron-app
pnpm install
```

### Running the Application (Development)

#### Quick Start (Recommended)

```bash
# One-click development server with both frontend and backend
cd electron-app
pnpm dev
```

#### Individual Components

```bash
# Streamlit WebUI
uv run zsim run

# FastAPI Backend
uv run zsim api

# Electron Desktop App (production build)
cd electron-app
pnpm build
```

**Note**: The `pnpm dev` command provides the most convenient development experience by:
- Automatically starting both the Vue.js frontend and FastAPI backend
- Forwarding all backend console output to the development terminal
- Providing hot reload for the frontend
- Enabling full debugging capabilities

### Testing

*   Unit tests in `tests/` directory.
*   API tests in `tests/api/`.
*   Fixtures defined in `tests/conftest.py`.
*   Uses pytest with asyncio support.

#### Run Tests

```bash
# Run the tests
uv run pytest
# Run the tests with coverage report
uv run pytest -v --cov=zsim --cov-report=html
```

## TODO List

Refer to the [develop guide](https://github.com/ZZZSimulator/ZSim/wiki/%E8%B4%A1%E6%8C%AE%E6%8C%87%E5%8D%97-Develop-Guide) for comprehensive details.

## Environment Variables

### FastAPI Backend
- `ZSIM_DISABLE_ROUTES` - Set to "1" to disable API routes (default: enabled)
- `ZSIM_IPC_MODE` - IPC communication mode: "auto", "uds", or "http" (default: "auto")
- `ZSIM_UDS_PATH` - UDS socket file path when using UDS mode (default: "/tmp/zsim_api.sock")
- `ZSIM_API_PORT` - API server port, set to 0 for automatic port selection (default: 0)
- `ZSIM_API_HOST` - API server host address (default: "127.0.0.1")

### IPC Mode Behavior
- **auto**: Uses uds on Unix like OS, and http on windows
- **uds**: Uses Unix Domain Socket for local communication (Unix like only)
- **http**: Uses HTTP/TCP for communication (default mode)