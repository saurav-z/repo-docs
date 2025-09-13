# ZSim: Zenless Zone Zero Battle Simulator & Damage Calculator

**Maximize your Zenless Zone Zero team's potential with ZSim, the automated battle simulator and damage calculator, empowering you to optimize your character builds and team compositions.**  [See the original repo](https://github.com/ZZZSimulator/ZSim)

<img src="./docs/img/横板logo成图.png" alt="ZSim Logo" width="400"/>

## Key Features

*   **Automated Battle Simulation:**  Simulates battles automatically based on your team's equipment and selected Action Priority List (APL).
*   **Comprehensive Damage Calculation:** Calculates total damage output, considering each character's equipment and weapon characteristics.
*   **User-Friendly Interface:** Provides an intuitive interface for editing agent equipment, selecting APLs, and running simulations.
*   **Visualized Results:** Generates reports with interactive charts and detailed tables for in-depth damage analysis.
*   **Character-Specific Damage Information:** Presents detailed damage breakdowns for each character in your team.
*   **APL Editing:**  Allows customization of Action Priority Lists to fine-tune your team's performance.
*   **Multiple Interface Options:** Includes a Streamlit-based Web UI, a FastAPI backend with REST API, a CLI, and an Electron-based Desktop App (Vue.js).

## Installation

### Prerequisites: Install UV Package Manager
ZSim uses the `uv` package manager for dependency management. Install it using one of the following methods:

*   **Using pip (if you have Python installed):**
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
*   **Earlier Windows versions:**
    ```powershell
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    ```
*  **Alternatively**, you can check the official installation guide: [https://docs.astral.sh/uv/getting-started/installation/](https://docs.astral.sh/uv/getting-started/installation/)

### Installing and Running ZSim

1.  **Navigate to the project directory** in your terminal.

2.  **Synchronize Dependencies:**
    ```bash
    uv sync
    ```

3.  **Run the ZSim Application:**
    ```bash
    uv run zsim run
    ```

## Development

### Key Components

*   **Simulation Engine:** Core logic in `zsim/simulator/`
*   **Web API:** FastAPI-based REST API in `zsim/api_src/`
*   **Web UI:** Streamlit-based interface in `zsim/webui.py` and new Vue.js + Electron desktop application in `electron-app/`
*   **CLI:** Command-line interface via `zsim/run.py`
*   **Database:** SQLite-based storage for character/enemy configurations
*   **Electron App:** Desktop application built with Vue.js and Electron that communicates with the FastAPI backend

### Build System

The project leverages a Make-based build system for efficient development and release management.

#### Available Make Targets

*   `make build`: Full build (clean + backend + electron)
*   `make backend`: Build backend API only
*   `make electron-build`: Build Electron desktop application only
*   `make dev`: Start frontend development server
*   `make clean`: Clean all build files
*   `make check`: Check dependencies
*   `make help`: Display help information

### Setup and Installation for Development

1.  **Install UV (as described above)**
2.  **For WebUI Development:**
    ```bash
    uv run zsim run
    ```
3.  **For FastAPI Backend:**
    ```bash
    uv run zsim api
    ```
4.  **For Electron App Development:**  Navigate to the `electron-app` directory and install Node.js dependencies:
    ```bash
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

#### Running Tests

*   Run all tests:
    ```bash
    uv run pytest
    ```
*   Run tests with coverage report:
    ```bash
    uv run pytest -v --cov=zsim --cov-report=html
    ```

## TODO List

Consult the [Develop Guide](https://github.com/ZZZSimulator/ZSim/wiki/%E8%B4%A1%E7%8C%AE%E6%8C%87%E5%8D%97-Develop-Guide) for the latest development plans and contributing guidelines.

## Environment Variables

### FastAPI Backend

*   `ZSIM_DISABLE_ROUTES`: Set to `"1"` to disable API routes (default: enabled)
*   `ZSIM_IPC_MODE`: IPC communication mode: `"auto"`, `"uds"`, or `"http"` (default: `"auto"`)
*   `ZSIM_UDS_PATH`: UDS socket file path when using UDS mode (default: `"/tmp/zsim_api.sock"`)
*   `ZSIM_API_PORT`: API server port, set to `0` for automatic port selection (default: `0`)
*   `ZSIM_API_HOST`: API server host address (default: `"127.0.0.1"`)

### IPC Mode Behavior

*   **auto:** Uses UDS on Unix-like OS, and HTTP on Windows.
*   **uds:** Uses Unix Domain Socket for local communication (Unix-like only).
*   **http:** Uses HTTP/TCP for communication (default mode).