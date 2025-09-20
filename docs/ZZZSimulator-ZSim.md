# ZSim: Zenless Zone Zero Battle Simulator & Damage Calculator

**Maximize your team's damage potential in Zenless Zone Zero with ZSim, a powerful and user-friendly battle simulator.** ([Original Repository](https://github.com/ZZZSimulator/ZSim))

## Key Features:

*   **Automated Simulation:**  Automatically simulates battles based on your team's equipment and a pre-defined Action Priority List (APL), eliminating manual skill sequence input.
*   **Comprehensive Damage Calculation:** Calculates total damage output, considering each character's weapons, equipment, and in-game mechanics.
*   **Visual Analysis & Reporting:** Generates insightful visual charts and detailed damage information for each character to optimize team composition and strategy.
*   **Agent Customization:** Allows you to edit and configure your agents' equipment to match your in-game builds.
*   **APL Customization:** Enables you to customize the Action Priority List (APL) for tailored simulations.

## Installation

### Prerequisites
1.  **Install UV:**  A fast Python package installer and resolver.

    *   **Using `pip`:**
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
        ```bash
        powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
        ```
    *   For detailed instructions, refer to the official UV documentation: [https://docs.astral.sh/uv/getting-started/installation/](https://docs.astral.sh/uv/getting-started/installation/)

### Installing and Running ZSim

1.  **Navigate:** Open your terminal in the project's directory.
2.  **Install Dependencies:**
    ```bash
    uv sync
    ```
3.  **Run the Simulator:**
    ```bash
    uv run zsim run
    ```

## Development

### Key Components:

*   **Simulation Engine:**  Located in `zsim/simulator/`, handles core battle logic.
*   **Web API:**  FastAPI-based REST API found in `zsim/api_src/`, for programmatic access.
*   **Web UI:**  A Streamlit-based interface in `zsim/webui.py` and a new Vue.js + Electron desktop application in `electron-app/`.
*   **CLI:**  Command-line interface via `zsim/run.py`.
*   **Database:** Uses SQLite for character and enemy configuration storage.
*   **Electron App:** Desktop application built with Vue.js and Electron, communicating with the FastAPI backend.

### Build System:

A comprehensive Make-based build system facilitates development, building, and release processes.

#### Available Make Targets:

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

### Setup and Installation for Development:

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

**Note**: The `pnpm dev` command simplifies development, starting both the Vue.js frontend and FastAPI backend, forwarding console output, enabling hot reload, and providing debugging capabilities.

### Testing Structure

*   **Unit Tests:**  Located in the `tests/` directory.
*   **API Tests:**  Located in the `tests/api/` directory.
*   **Fixtures:** Defined in `tests/conftest.py`.
*   Uses `pytest` with `asyncio` support.

```bash
# Run the tests
uv run pytest
# Run the tests with coverage report
uv run pytest -v --cov=zsim --cov-report=html
```

## TODO List

Refer to the [development guide](https://github.com/ZZZSimulator/ZSim/wiki/%E8%B4%A1%E7%8C%AE%E6%8C%87%E5%8D%97-Develop-Guide) for detailed information.

## Environment Variables

### FastAPI Backend

*   `ZSIM_DISABLE_ROUTES`: Disable API routes.  Set to `"1"` (default: enabled).
*   `ZSIM_IPC_MODE`: IPC communication mode: `"auto"`, `"uds"`, or `"http"` (default: `"auto"`).
*   `ZSIM_UDS_PATH`: UDS socket file path when using UDS mode (default: `"/tmp/zsim_api.sock"`).
*   `ZSIM_API_PORT`: API server port, set to `0` for automatic port selection (default: `0`).
*   `ZSIM_API_HOST`: API server host address (default: `"127.0.0.1"`).

### IPC Mode Behavior

*   **auto:** Uses UDS on Unix-like OS and HTTP on Windows.
*   **uds:**  Uses Unix Domain Socket for local communication (Unix-like only).
*   **http:** Uses HTTP/TCP for communication (default).