# ZSim: Zenless Zone Zero Battle Simulator & Damage Calculator

**Maximize your team's potential in Zenless Zone Zero with ZSim, an automated battle simulator and damage calculator that provides detailed insights for optimal gameplay.**  [View the original repository](https://github.com/ZZZSimulator/ZSim)

## Key Features

*   **Automated Battle Simulation:**  Simulates battles based on your team composition and equipment, eliminating the need for manual skill sequence input (unless you need sequence mode, just let us know!).
*   **Damage Calculation:** Calculates total damage output, factoring in each character's weapons, equipment, and buffs.
*   **Visual Reporting:** Generates easy-to-understand visual charts and tables summarizing damage output and character performance.
*   **Agent Equipment & APL Editing:**  Allows you to customize agent equipment and edit Action Priority Lists (APLs) for precise control.

## Installation

### Prerequisites
*   Python (if you don't have python)
*   Install UV package manager (recommended):

    *   **Using pip (if you have python installed):**
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
    *   **Older Windows versions:**
        ```powershell
        powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
        ```
    *   Refer to the official UV installation guide for more information: [https://docs.astral.sh/uv/getting-started/installation/](https://docs.astral.sh/uv/getting-started/installation/)

### Installation and Running ZSim
1.  Open your terminal and navigate to the project directory.
2.  Install project dependencies:
    ```bash
    uv sync
    ```
3.  Run the simulator:
    ```bash
    uv run zsim run
    ```

## Development

### Key Components
*   **Simulation Engine:** `zsim/simulator/` - Core battle simulation logic.
*   **Web API:** `zsim/api_src/` - FastAPI-based REST API.
*   **Web UI:** `zsim/webui.py` & `electron-app/` - Streamlit and Vue.js/Electron based interfaces.
*   **CLI:** `zsim/run.py` - Command-line interface.
*   **Database:** SQLite-based configuration storage.
*   **Electron App:** Desktop application built with Vue.js and Electron.

### Build System

Uses a Make-based build system.

#### Build Targets

```bash
make build          # Full build (clean + backend + electron)
make backend        # Build backend API only
make electron-build # Build Electron desktop application only
```

#### Development

```bash
make dev                # Start frontend development server
make clean              # Clean all build files
make check              # Check dependencies
```

#### Utilities

```bash
make help               # Display help information
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

**Note:** The `pnpm dev` command offers the best development experience.

### Testing Structure

*   Unit tests in `tests/`
*   API tests in `tests/api/`
*   Fixtures in `tests/conftest.py`
*   Uses pytest with asyncio support

```bash
# Run the tests
uv run pytest
# Run the tests with coverage report
uv run pytest -v --cov=zsim --cov-report=html
```

## TODO List

See the [development guide](https://github.com/ZZZSimulator/ZSim/wiki/%E8%B4%A1%E7%8C%AE%E6%8C%87%E5%8D%97-Develop-Guide) for details.

## Environment Variables

### FastAPI Backend
*   `ZSIM_DISABLE_ROUTES` (default: enabled)
*   `ZSIM_IPC_MODE` (default: "auto")
*   `ZSIM_UDS_PATH` (default: "/tmp/zsim_api.sock")
*   `ZSIM_API_PORT` (default: 0)
*   `ZSIM_API_HOST` (default: "127.0.0.1")

### IPC Mode Behavior
*   **auto**: Uses uds on Unix like OS, and http on windows
*   **uds**: Uses Unix Domain Socket for local communication (Unix like only)
*   **http**: Uses HTTP/TCP for communication (default mode)