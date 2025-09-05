# ZSim: Zenless Zone Zero (ZZZ) Battle Simulator and Damage Calculator

**Optimize your team composition in Zenless Zone Zero with ZSim, the automated battle simulator and damage calculator!** ([Original Repository](https://github.com/ZZZSimulator/ZSim))

<img src="./docs/img/横板logo成图.png" alt="ZSim Logo" width="400">

ZSim is your go-to tool for analyzing team damage in Zenless Zone Zero (ZZZ), the action game from Hoyoverse.  It automatically simulates battles, eliminating the need for manual skill sequences. Simply equip your agents, select an Action Priority List (APL), and run the simulation.

## Key Features:

*   **Automated Battle Simulation:**  Focus on team building, not micromanagement.
*   **Comprehensive Damage Calculation:**  Analyze total team damage with weapon and equipment considerations.
*   **Visual Reporting:**  Understand your team's performance with interactive charts and detailed tables.
*   **Agent Customization:**  Edit agent equipment to tailor your team strategy.
*   **APL Integration:**  Customize character actions with APL code.

## Installation

### Prerequisites:
*   **UV Package Manager:** Install UV using one of the following methods. UV is a package manager that is needed to run this project.

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

*For detailed UV installation instructions, consult the official documentation: [https://docs.astral.sh/uv/getting-started/installation/](https://docs.astral.sh/uv/getting-started/installation/)*

### ZSim Installation and Running

1.  **Clone or Download:** Get the latest source code from the [release page](https://github.com/ZZZSimulator/ZSim/releases) or clone the repository using Git:

    ```bash
    git clone https://github.com/ZZZSimulator/ZSim.git
    cd ZSim
    ```

2.  **Install Dependencies and Run:**  Navigate to the project directory in your terminal and run:

    ```bash
    uv sync
    uv run zsim run
    ```

## Development

ZSim's development involves several key components:

*   **Simulation Engine:** Core battle logic in `zsim/simulator/`.
*   **Web API:** FastAPI-based REST API in `zsim/api_src/`.
*   **Web UI:** Streamlit-based interface (`webui.py`) and a new Vue.js + Electron desktop application (`electron-app/`).
*   **CLI:** Command-line interface via `zsim/run.py`.
*   **Database:** SQLite-based storage for character and enemy data.
*   **Electron App:** Desktop application built with Vue.js and Electron, communicating with the FastAPI backend.

### Build System

ZSim uses a comprehensive Make-based build system for managing development, building, and release processes.

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

ZSim uses a pytest-based testing structure with the following organization:

*   **Unit Tests:** In the `tests/` directory.
*   **API Tests:** In the `tests/api/` directory.
*   **Fixtures:** Defined in `tests/conftest.py`.
*   **Framework:** Uses pytest with asyncio support.

```bash
# Run the tests
uv run pytest
# Run the tests with coverage report
uv run pytest -v --cov=zsim --cov-report=html
```

## TODO List

See the [develop guide](https://github.com/ZZZSimulator/ZSim/wiki/%E8%B4%A1%E7%8C%AE%E6%8C%87%E5%8D%97-Develop-Guide) for detailed development tasks.

## Environment Variables

### FastAPI Backend

*   `ZSIM_DISABLE_ROUTES` - Disable API routes (default: enabled)
*   `ZSIM_IPC_MODE` - Inter-Process Communication mode: "auto", "uds", or "http" (default: "auto")
*   `ZSIM_UDS_PATH` - UDS socket file path when using UDS mode (default: "/tmp/zsim_api.sock")
*   `ZSIM_API_PORT` - API server port, set to 0 for automatic port selection (default: 0)
*   `ZSIM_API_HOST` - API server host address (default: "127.0.0.1")

### IPC Mode Behavior

*   **auto:** Uses UDS on Unix-like OS and HTTP on Windows.
*   **uds:** Uses Unix Domain Socket for local communication (Unix-like OS only).
*   **http:** Uses HTTP/TCP for communication (default mode).