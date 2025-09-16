# ZSim: Zenless Zone Zero Battle Simulator & Damage Calculator

**Maximize your team's potential in Zenless Zone Zero with ZSim, the automated battle simulator and damage calculator.** ([View the Original Repository](https://github.com/ZZZSimulator/ZSim))

## About ZSim

ZSim is a powerful tool designed to simulate battles and calculate damage output for characters in the action game Zenless Zone Zero (ZZZ). By automatically simulating actions based on preset Action Priority Lists (APLs), ZSim eliminates the need for manual skill sequence input, providing a user-friendly interface to analyze team compositions and optimize your strategy.

## Key Features

*   **Automated Battle Simulation:**  Simulates battles based on predefined APLs, saving you time and effort.
*   **Comprehensive Damage Calculation:** Calculates total damage output, considering each character's equipment and weapon characteristics.
*   **Visual Reporting:** Generates visual charts and tables for easy analysis of results.
*   **Team Customization:**  Edit your agents' equipment and customize APLs to fine-tune your strategy.
*   **Detailed Damage Breakdown:** Provides in-depth damage information for each character in your team.

## Installation

### Prerequisites

1.  **UV Package Manager:** If you haven't already, install UV using one of the following commands in your terminal:

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
    Or check the official installation guide: [https://docs.astral.sh/uv/getting-started/installation/](https://docs.astral.sh/uv/getting-started/installation/)

### Installing and Running ZSim

1.  **Navigate to Project Directory:** Open your terminal and navigate to the project directory where you've downloaded the ZSim source code.
2.  **Sync Dependencies:** Run `uv sync` to install the project dependencies.
3.  **Run the Application:** Execute `uv run zsim run` to start the ZSim application.

## Development

### Key Components

*   **Simulation Engine:** Core battle simulation logic resides in `zsim/simulator/`.
*   **Web API:**  FastAPI-based REST API in `zsim/api_src/`.
*   **Web UI:**  Streamlit-based interface in `zsim/webui.py` and a new Vue.js + Electron desktop application in `electron-app/`.
*   **CLI:** Command-line interface via `zsim/run.py`.
*   **Database:** SQLite database for character and enemy configurations.
*   **Electron App:** Desktop application built with Vue.js and Electron.

### Build System

The project uses a Make-based build system for managing development, building, and release processes.

#### Available Make Targets

```bash
# Build components
make build              # Full build (clean + backend + electron)
make backend            # Build backend API only
make electron-build     # Build Electron desktop application only

# Development
make dev                # Start frontend development server
make clean              # Clean all build files
make check              # Check dependencies

# Utilities
make help                # Display help information
```

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
pnpm install
```

### Running the Application

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

## TODO List

Refer to the [Develop Guide](https://github.com/ZZZSimulator/ZSim/wiki/%E8%B4%A1%E7%8C%AE%E6%8C%87%E5%8D%97-Develop-Guide) for more details.

## Environment Variables

### FastAPI Backend

*   `ZSIM_DISABLE_ROUTES`:  Set to "1" to disable API routes (default: enabled)
*   `ZSIM_IPC_MODE`: IPC communication mode: "auto", "uds", or "http" (default: "auto")
*   `ZSIM_UDS_PATH`: UDS socket file path when using UDS mode (default: "/tmp/zsim_api.sock")
*   `ZSIM_API_PORT`: API server port, set to 0 for automatic port selection (default: 0)
*   `ZSIM_API_HOST`: API server host address (default: "127.0.0.1")

### IPC Mode Behavior

*   **auto**: Uses uds on Unix like OS, and http on windows
*   **uds**: Uses Unix Domain Socket for local communication (Unix like only)
*   **http**: Uses HTTP/TCP for communication (default mode)