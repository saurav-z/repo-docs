# ZSim: Zenless Zone Zero Battle Simulator & Damage Calculator

**Maximize your team's performance in Zenless Zone Zero with ZSim, a fully automated battle simulator and damage calculator!** ([Original Repository](https://github.com/ZZZSimulator/ZSim))

## Key Features

*   **Automated Simulation:** No manual skill sequence input is required.
*   **Team Damage Calculation:** Computes total damage output based on team composition and equipment.
*   **Visual Reports:** Generates charts and tables for easy analysis.
*   **Character & Equipment Editor:** Allows customization of agent equipment.
*   **APL Code Editing:** Enables users to customize action priority lists.

## Installation

### Prerequisites

You need to install `UV`, a fast Python package manager.

**Install UV:**

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

### Install & Run ZSim

1.  **Clone or Download:** Get the latest source code from the release page or use `git clone`.
2.  **Navigate:** Open your terminal in the project directory.
3.  **Install Dependencies & Run:**

    ```bash
    uv sync
    uv run zsim run
    ```

## Development

### Key Components

*   **Simulation Engine:** Core logic within `zsim/simulator/` for battle simulation.
*   **Web API:** FastAPI-based REST API in `zsim/api_src/`.
*   **Web UI:** Streamlit-based interface and new Vue.js + Electron desktop application.
*   **CLI:** Command-line interface via `zsim/run.py`.
*   **Database:** SQLite-based storage for character and enemy data.
*   **Electron App:** Desktop app (Vue.js & Electron) communicating with the FastAPI backend.

### Build System

Uses a Make-based build system.

**Available Make Targets:**

*   `make build`: Full build (clean + backend + electron)
*   `make backend`: Build backend API only
*   `make electron-build`: Build Electron desktop application only
*   `make dev`: Start frontend development server
*   `make clean`: Clean all build files
*   `make check`: Check dependencies
*   `make help`: Display help information

### Setup & Installation

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

**Quick Start (Recommended):**

```bash
# One-click development server with both frontend and backend
cd electron-app
pnpm dev
```

**Individual Components:**

```bash
# Streamlit WebUI
uv run zsim run

# FastAPI Backend
uv run zsim api

# Electron Desktop App (production build)
cd electron-app
pnpm build
```

**Note:** `pnpm dev` provides the best development experience.

### Testing

*   Unit tests in `tests/`
*   API tests in `tests/api/`
*   Fixtures defined in `tests/conftest.py`
*   Uses pytest with asyncio support

**Run Tests:**

```bash
uv run pytest
# Run the tests with coverage report
uv run pytest -v --cov=zsim --cov-report=html
```

## Environment Variables

### FastAPI Backend

*   `ZSIM_DISABLE_ROUTES`: Disable API routes ("1" to disable, default: enabled)
*   `ZSIM_IPC_MODE`: IPC communication mode ("auto", "uds", "http", default: "auto")
*   `ZSIM_UDS_PATH`: UDS socket file path (default: "/tmp/zsim_api.sock")
*   `ZSIM_API_PORT`: API server port (set to 0 for auto, default: 0)
*   `ZSIM_API_HOST`: API server host address (default: "127.0.0.1")

### IPC Mode Behavior

*   **auto**: Uses UDS on Unix-like OS, HTTP on Windows.
*   **uds**: Unix Domain Socket (Unix-like only).
*   **http**: HTTP/TCP.