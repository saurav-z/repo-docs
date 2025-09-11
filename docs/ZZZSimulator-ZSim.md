# ZSim: Zenless Zone Zero Battle Simulator and Damage Calculator

**Unleash the power of strategic team building with ZSim, the fully automated damage calculator for Zenless Zone Zero.** ([Original Repository](https://github.com/ZZZSimulator/ZSim))

<p align="center">
  <img src="./docs/img/横板logo成图.png" alt="ZSim Logo" />
</p>

ZSim is a powerful tool designed to help you optimize your team compositions and maximize damage output in Zenless Zone Zero (ZZZ).  It automatically simulates battles, taking into account character equipment and action priority lists (APLs), to provide you with detailed insights.

## Key Features

*   **Automated Battle Simulation:**  No need for manual skill sequence input. The simulator runs automatically based on preset APLs.
*   **Comprehensive Damage Calculation:**  Calculates total damage output for your team, considering character stats, weapon effects, and equipment.
*   **User-Friendly Interface:**  Provides an intuitive interface for editing agent equipment and APLs.
*   **Visual Reporting:** Generates detailed damage information presented in clear charts and tables.
*   **Flexible APL Editing:**  Allows you to customize action priority lists to test different strategies.

## Installation

ZSim can be installed easily using `uv` and then running the application.

### Prerequisites: Install `uv` (if you haven't already)

Choose your operating system:

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

1.  Open a terminal in the project directory.
2.  Run the following commands:

    ```bash
    uv sync
    uv run zsim run
    ```

## Development

ZSim's development environment includes the following key components:

*   **Simulation Engine:** Core battle simulation logic located in `zsim/simulator/`.
*   **Web API:** A FastAPI-based REST API in `zsim/api_src/` for programmatic access.
*   **Web UI:** A Streamlit-based interface in `zsim/webui.py` and a new Vue.js + Electron desktop application in `electron-app/`.
*   **CLI:** Command-line interface via `zsim/run.py`.
*   **Database:** SQLite-based storage for character and enemy configurations.
*   **Electron App:** Desktop application built with Vue.js and Electron that communicates with the FastAPI backend.

### Build System

A Make-based build system streamlines development, building, and release processes.

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

**Note:**  The `pnpm dev` command offers the best development experience, automatically starting both frontend and backend, forwarding console output, and enabling hot reloading and full debugging.

### Testing Structure

*   Unit tests located in the `tests/` directory.
*   API tests located in `tests/api/`.
*   Fixtures are defined in `tests/conftest.py`.
*   Uses pytest with asyncio support.

```bash
# Run the tests
uv run pytest
# Run the tests with coverage report
uv run pytest -v --cov=zsim --cov-report=html
```

## TODO List

For more details on future development, consult the [Develop Guide](https://github.com/ZZZSimulator/ZSim/wiki/%E8%B4%A1%E7%8C%AE%E6%8C%87%E5%8D%97-Develop-Guide).

## Environment Variables

### FastAPI Backend

*   `ZSIM_DISABLE_ROUTES`:  Set to "1" to disable API routes (default: enabled).
*   `ZSIM_IPC_MODE`:  IPC communication mode: "auto", "uds", or "http" (default: "auto").
*   `ZSIM_UDS_PATH`:  UDS socket file path when using UDS mode (default: "/tmp/zsim_api.sock").
*   `ZSIM_API_PORT`:  API server port; set to 0 for automatic port selection (default: 0).
*   `ZSIM_API_HOST`:  API server host address (default: "127.0.0.1").

### IPC Mode Behavior

*   **auto:** Uses UDS on Unix-like OS and HTTP on Windows.
*   **uds:** Uses Unix Domain Socket for local communication (Unix-like only).
*   **http:** Uses HTTP/TCP for communication (default mode).