# ZSim: Zenless Zone Zero Battle Simulator & Damage Calculator

**Maximize your team's damage potential in Zenless Zone Zero with ZSim, the automated battle simulator and damage calculator.** ([Original Repository](https://github.com/ZZZSimulator/ZSim))

## Key Features

*   **Automated Simulation:** Automatically simulates battles based on Action Priority Lists (APL), eliminating the need for manual skill sequence input.
*   **Comprehensive Damage Calculation:** Calculates total damage output based on team composition, character equipment, and weapon characteristics.
*   **Visual Reporting:** Generates detailed reports with visual charts and tables for easy analysis.
*   **Agent Customization:** Allows you to edit agents' equipment to test out different builds.
*   **APL Editing:** Provides the ability to edit APL code for advanced strategy customization.

## Installation

### Prerequisites:

*   **UV (Package Manager):** Before installing ZSim, install UV:

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
        winget install --id=astral-sh.uv -e
        ```
    *   **Older Windows versions:**
        ```powershell
        powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
        ```
        You can also refer to the official UV installation guide: [https://docs.astral.sh/uv/getting-started/installation/](https://docs.astral.sh/uv/getting-started/installation/)

### Installing ZSim

1.  **Clone or Download:** Obtain the latest source code from the [release page](https://github.com/ZZZSimulator/ZSim/releases) or by using `git clone`.
2.  **Navigate to Project Directory:** Open your terminal and navigate to the project's root directory.
3.  **Install Dependencies and Run:**
    ```bash
    uv sync
    uv run zsim run
    ```

## Development

### Key Components

*   **Simulation Engine:** Core logic located in `zsim/simulator/`.
*   **Web API:** FastAPI-based REST API in `zsim/api_src/`.
*   **Web UI:** Streamlit-based interface in `zsim/webui.py` and a new Vue.js + Electron desktop application in `electron-app/`.
*   **CLI:** Command-line interface via `zsim/run.py`.
*   **Database:** SQLite-based storage for character/enemy configurations.
*   **Electron App:** Desktop application (built with Vue.js and Electron) that communicates with the FastAPI backend.

### Build System

The project utilizes a comprehensive Make-based build system.

#### Available Make Targets

*   `make build`: Full build (clean + backend + electron)
*   `make backend`: Build backend API only
*   `make electron-build`: Build Electron desktop application only
*   `make dev`: Start frontend development server
*   `make clean`: Clean all build files
*   `make check`: Check dependencies
*   `make help`: Display help information

### Setup and Installation (Development)

```bash
# Install UV package manager first (if not already installed)
uv sync

# For WebUI development
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

**Note:** The `pnpm dev` command offers the best development experience by:
* Automatically starting both Vue.js frontend and FastAPI backend.
* Forwarding all backend console output to the development terminal.
* Providing hot reload for the frontend.
* Enabling full debugging capabilities.

### Testing Structure

*   Unit tests are located in the `tests/` directory.
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

For further development information, refer to the [develop guide](https://github.com/ZZZSimulator/ZSim/wiki/%E8%B4%A1%E7%8C%AE%E6%8C%87%E5%8D%97-Develop-Guide).

## Environment Variables

### FastAPI Backend

*   `ZSIM_DISABLE_ROUTES`: Set to "1" to disable API routes (default: enabled).
*   `ZSIM_IPC_MODE`: IPC communication mode: "auto", "uds", or "http" (default: "auto").
*   `ZSIM_UDS_PATH`: UDS socket file path when using UDS mode (default: "/tmp/zsim_api.sock").
*   `ZSIM_API_PORT`: API server port, set to 0 for automatic port selection (default: 0).
*   `ZSIM_API_HOST`: API server host address (default: "127.0.0.1").

### IPC Mode Behavior

*   **auto:** Uses UDS on Unix-like OS and HTTP on Windows.
*   **uds:** Uses Unix Domain Socket for local communication (Unix-like only).
*   **http:** Uses HTTP/TCP for communication (default mode).