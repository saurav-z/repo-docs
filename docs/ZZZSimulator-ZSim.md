# ZSim: Zenless Zone Zero Battle Simulator & Damage Calculator

**Maximize your team's potential in Zenless Zone Zero with ZSim, the fully automatic battle simulator and damage calculator!**

[View the original repository on GitHub](https://github.com/ZZZSimulator/ZSim)

ZSim provides a user-friendly platform to analyze team compositions, optimize equipment, and predict damage output for Zenless Zone Zero (ZZZ). It automatically simulates battles based on your agent configurations and a selected Action Priority List (APL), generating detailed reports for in-depth analysis.

## Key Features:

*   **Automated Battle Simulation:** Automatically simulates battles based on pre-set APLs, removing the need for manual skill sequence input.
*   **Comprehensive Damage Calculation:** Calculates total damage output, taking into account character, weapon, and equipment stats.
*   **Visual Reporting:** Generates interactive charts and tables for clear and concise damage information.
*   **Agent Customization:** Allows users to edit agent equipment to create their best team builds.
*   **APL Editing:**  Offers APL customization to fine-tune battle strategies.

## Installation Guide

### Prerequisites

Ensure you have Python installed on your system.

### Installing UV (Package Manager)

ZSim uses the UV package manager. Follow the appropriate installation instructions:

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

### Installing and Running ZSim

1.  **Navigate to the project directory:** Open your terminal and navigate to the directory where you have downloaded or cloned the ZSim project.
2.  **Install Dependencies:**
    ```bash
    uv sync
    ```
3.  **Run ZSim:**
    *   **Web UI:**
        ```bash
        uv run zsim run
        ```

## Development

### Key Components

ZSim is built with the following components:

*   **Simulation Engine:** `zsim/simulator/` - Handles battle simulation logic.
*   **Web API:** `zsim/api_src/` - FastAPI-based REST API.
*   **Web UI:** `zsim/webui.py` (Streamlit) and `electron-app/` (Vue.js + Electron).
*   **CLI:** `zsim/run.py` - Command-line interface.
*   **Database:** SQLite-based database for storing character and enemy data.
*   **Electron App:** Desktop application (Vue.js and Electron) that interacts with the FastAPI backend.

### Build System

The project utilizes a Make-based build system.

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

### Setup and Installation (Development)

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

**Note:** `pnpm dev` offers the best development experience.

### Testing Structure

*   Unit tests are located in the `tests/` directory.
*   API tests are in `tests/api/`.
*   Fixtures are defined in `tests/conftest.py`.
*   pytest with asyncio support is used for testing.

```bash
# Run the tests
uv run pytest
# Run the tests with coverage report
uv run pytest -v --cov=zsim --cov-report=html
```

## Environment Variables

### FastAPI Backend

*   `ZSIM_DISABLE_ROUTES`: Set to "1" to disable API routes (default: enabled)
*   `ZSIM_IPC_MODE`: IPC communication mode: "auto", "uds", or "http" (default: "auto")
*   `ZSIM_UDS_PATH`: UDS socket file path when using UDS mode (default: "/tmp/zsim_api.sock")
*   `ZSIM_API_PORT`: API server port, set to 0 for automatic port selection (default: 0)
*   `ZSIM_API_HOST`: API server host address (default: "127.0.0.1")

### IPC Mode Behavior

*   **auto**: Uses uds on Unix like OS, and http on windows
*   **uds**: Uses Unix Domain Socket for local communication (Unix like only)
*   **http**: Uses HTTP/TCP for communication (default mode)