# ZSim: Zenless Zone Zero Battle Simulator & Damage Calculator

**Optimize your team's performance in Zenless Zone Zero with ZSim, a powerful and automated battle simulator and damage calculator.**

[View the original repository on GitHub](https://github.com/ZZZSimulator/ZSim)

## Key Features

*   **Automated Battle Simulation:** Automatically simulates battles based on your team composition and equipment.
*   **Detailed Damage Calculation:** Provides precise damage output calculations for each character.
*   **Visual Reporting:** Generates comprehensive reports with visual charts for easy analysis.
*   **Equipment & APL Editing:** Allows customization of character equipment and APL (Action Priority List) for tailored simulations.
*   **User-Friendly Interface:** Simple interface for selecting agents, editing equipment, and running simulations.
*   **Supports Action Priority Lists (APLs):** Allows for the use of preset APLs for battle simulations.

## Installation

### Prerequisites

1.  **Install `uv` (if you haven't already):** A fast Python package and virtual environment manager

    *   **Using pip (if Python is installed):**

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
    *   **On older versions of Windows:**

        ```powershell
        powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
        ```

    *For more detailed installation guidance, see the official documentation: [https://docs.astral.sh/uv/getting-started/installation/](https://docs.astral.sh/uv/getting-started/installation/)*

### Installing and Running ZSim

1.  **Navigate to the Project Directory:** Open your terminal and change directory to the location where you downloaded the ZSim project.

2.  **Install Dependencies and Run:**

    ```bash
    uv sync
    uv run zsim run
    ```

## Development

### Key Components

*   **Simulation Engine:** Core logic located in `zsim/simulator/` for handling battle simulations.
*   **Web API:** FastAPI-based REST API within `zsim/api_src/` for programmatic access.
*   **Web UI:** Streamlit-based interface in `zsim/webui.py` and a new Vue.js + Electron desktop application in `electron-app/`.
*   **CLI:** Command-line interface available via `zsim/run.py`.
*   **Database:** SQLite-based storage for character and enemy configurations.
*   **Electron App:** A desktop application built with Vue.js and Electron that communicates with the FastAPI backend.

### Build System

The project uses a Make-based build system for managing development, build, and release processes.

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

### Setup and Installation for Development

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

*   Unit tests in the `tests/` directory
*   API tests in `tests/api/`
*   Fixtures are defined in `tests/conftest.py`
*   Uses pytest with asyncio support

```bash
# Run the tests
uv run pytest
# Run the tests with coverage report
uv run pytest -v --cov=zsim --cov-report=html
```

## TODO List

For further details, please refer to the [Development Guide](https://github.com/ZZZSimulator/ZSim/wiki/%E8%B4%A1%E7%8C%AE%E6%8C%87%E5%8D%97-Develop-Guide).

## Environment Variables

### FastAPI Backend

*   `ZSIM_DISABLE_ROUTES` - Set to "1" to disable API routes (default: enabled)
*   `ZSIM_IPC_MODE` - IPC communication mode: "auto", "uds", or "http" (default: "auto")
*   `ZSIM_UDS_PATH` - UDS socket file path when using UDS mode (default: "/tmp/zsim_api.sock")
*   `ZSIM_API_PORT` - API server port, set to 0 for automatic port selection (default: 0)
*   `ZSIM_API_HOST` - API server host address (default: "127.0.0.1")

### IPC Mode Behavior

*   **auto:** Uses uds on Unix like OS, and http on windows
*   **uds:** Uses Unix Domain Socket for local communication (Unix like only)
*   **http:** Uses HTTP/TCP for communication (default mode)