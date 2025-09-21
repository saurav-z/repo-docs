# ZSim: Zenless Zone Zero Battle Simulator and Damage Calculator

**Maximize your team's damage output in Zenless Zone Zero with ZSim, a fully automated battle simulator!**  Check out the original repo [here](https://github.com/ZZZSimulator/ZSim).

## Key Features

*   **Automated Battle Simulation:** No manual skill sequence input required.
*   **Comprehensive Damage Calculation:** Calculates total damage output based on team composition, character equipment, and weapon characteristics.
*   **Visual Reporting:** Generates informative charts and tables to analyze damage results.
*   **Agent Customization:** Allows editing of agent equipment.
*   **Action Priority List (APL) Editor:** Edit APL code to customize battle behavior.

## Installation

### Prerequisites

Before installing ZSim, ensure you have the following installed:

*   **Python:** (if not already installed)
*   **UV Package Manager:** This is the recommended package manager for ZSim. Install it using one of the following methods:

    *   **Using pip (if you have Python installed):**
        ```bash
        pip install uv
        ```

    *   **macOS or Linux:**
        ```bash
        curl -LsSf https://astral.sh/uv/install.sh | sh
        ```

    *   **Windows (Windows 11 24H2 or later):**
        ```bash
        winget install --id=astral-sh.uv  -e
        ```

    *   **Windows (Older versions):**
        ```powershell
        powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
        ```

### ZSim Installation and Running

1.  **Clone or download the source code:**
    *   Download the latest source code from the [releases page](https://github.com/ZZZSimulator/ZSim/releases).
    *   Alternatively, use `git clone`:

        ```bash
        git clone <repository_url>  # Replace <repository_url> with the ZSim repository URL
        cd ZSim  # Navigate into the project directory
        ```

2.  **Install dependencies and run:**
    ```bash
    uv sync
    uv run zsim run
    ```

## Development

ZSim utilizes a modular architecture:

*   **Simulation Engine:** Located in `zsim/simulator/` for core battle logic.
*   **Web API:**  A FastAPI-based REST API in `zsim/api_src/`.
*   **Web UI:**  A Streamlit-based interface in `zsim/webui.py` and a Vue.js + Electron desktop application.
*   **CLI:** Command-line interface in `zsim/run.py`.
*   **Database:** SQLite-based storage for character and enemy data.
*   **Electron App:** A desktop application built with Vue.js and Electron, interacting with the FastAPI backend.

### Build System

The project uses a Make-based build system for managing development, building, and release processes.

#### Available Make Targets

*   `make build`: Full build (clean + backend + electron)
*   `make backend`: Build backend API only
*   `make electron-build`: Build Electron desktop application only
*   `make dev`: Start frontend development server
*   `make clean`: Clean all build files
*   `make check`: Check dependencies
*   `make help`: Display help information

### Setup and Installation

```bash
# Install UV package manager first (if you haven't already)
uv sync

# For WebUI development
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

**Note:** `pnpm dev` provides the best development experience by automatically starting the frontend and backend, forwarding console output, hot reloading, and full debugging capabilities.

### Testing Structure

*   Unit tests are in the `tests/` directory.
*   API tests are in `tests/api/`.
*   Fixtures are defined in `tests/conftest.py`.
*   Uses pytest with asyncio support.

```bash
# Run the tests
uv run pytest

# Run the tests with coverage report
uv run pytest -v --cov=zsim --cov-report=html
```

## TODO List

Refer to the [develop guide](https://github.com/ZZZSimulator/ZSim/wiki/%E8%B4%A1%E7%8C%AE%E6%8C%87%E5%8D%97-Develop-Guide) for more details.

## Environment Variables

### FastAPI Backend

*   `ZSIM_DISABLE_ROUTES`: Set to "1" to disable API routes (default: enabled)
*   `ZSIM_IPC_MODE`: IPC communication mode: "auto", "uds", or "http" (default: "auto")
*   `ZSIM_UDS_PATH`: UDS socket file path when using UDS mode (default: "/tmp/zsim_api.sock")
*   `ZSIM_API_PORT`: API server port, set to 0 for automatic port selection (default: 0)
*   `ZSIM_API_HOST`: API server host address (default: "127.0.0.1")

### IPC Mode Behavior

*   **auto:** Uses UDS on Unix-like OS, and HTTP on Windows.
*   **uds:** Uses Unix Domain Socket for local communication (Unix-like only).
*   **http:** Uses HTTP/TCP for communication (default mode).