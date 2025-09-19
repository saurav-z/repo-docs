# ZSim: Zenless Zone Zero Battle Simulator & Damage Calculator

**Maximize your team's potential in Zenless Zone Zero with ZSim, an automated battle simulator and damage calculator.** ([Original Repo](https://github.com/ZZZSimulator/ZSim))

## Key Features

*   **Automated Battle Simulation:** Automatically simulates battles based on your team composition and Action Priority Lists (APLs).
*   **Comprehensive Damage Calculation:** Calculates total damage output, considering character equipment, weapons, and team synergies.
*   **Visual Reporting:** Generates easy-to-understand charts and tables to visualize damage and performance.
*   **Character Customization:** Allows you to edit and customize your agents' equipment.
*   **APL Editing:** Provides the ability to customize APL (Action Priority List) to further refine the simulation.
*   **User-Friendly Interface:** Features a Streamlit-based web UI and a new Vue.js + Electron desktop application for easy use.

## Installation

### Prerequisites

Ensure you have Python installed. Then, install the `uv` package manager:

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

1.  **Clone the repository** or download the latest source code from the [release page](https://github.com/ZZZSimulator/ZSim/releases).
2.  **Navigate** to the project directory in your terminal.
3.  **Install dependencies:**

    ```bash
    uv sync
    ```
4.  **Run the Simulator:**

    ```bash
    uv run zsim run
    ```

## Development

ZSim utilizes a comprehensive Make-based build system.

### Key Components

*   **Simulation Engine:** Core logic in `zsim/simulator/` for battle simulations.
*   **Web API:** FastAPI-based REST API in `zsim/api_src/` for programmatic access.
*   **Web UI:** Streamlit-based interface in `zsim/webui.py` and a new Vue.js + Electron desktop application in `electron-app/`.
*   **CLI:** Command-line interface via `zsim/run.py`.
*   **Database:** SQLite-based storage for character/enemy configurations.
*   **Electron App:** Desktop application built with Vue.js and Electron that communicates with the FastAPI backend.

### Build System

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

### Testing

Uses pytest with asyncio support.
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

## Contribute

See the [develop guide](https://github.com/ZZZSimulator/ZSim/wiki/%E8%B4%A1%E7%8C%AE%E6%8C%87%E5%8D%97-Develop-Guide) for details on how to contribute.