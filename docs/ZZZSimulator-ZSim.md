# ZSim: Zenless Zone Zero Battle Simulator & Damage Calculator

**Maximize your team's damage potential in Zenless Zone Zero with ZSim, a powerful and user-friendly battle simulator.** ([View on GitHub](https://github.com/ZZZSimulator/ZSim))

## Key Features

*   **Automatic Battle Simulation:** No need to manually input skill sequences. ZSim automatically simulates battles based on your team composition and equipment.
*   **Detailed Damage Calculation:** Calculate the total damage output of your team, taking into account character weapons, and equipment.
*   **Visual Data Analysis:** Generate intuitive charts and tables to visualize damage output and performance metrics.
*   **Character & Equipment Customization:** Easily edit agent equipment and experiment with different builds.
*   **Action Priority List (APL) Editing:** Customize battle strategies by editing the APL code.

## Installation

### Prerequisites

1.  **UV Package Manager:**

    Install UV, a fast Python package manager, using one of the following methods:

    ```bash
    # Using pip if you have Python installed:
    pip install uv
    ```

    ```bash
    # On macOS or Linux:
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

    ```bash
    # On Windows 11 24H2 or later:
    winget install --id=astral-sh.uv  -e
    ```

    ```bash
    # On lower version of Windows:
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    ```

    For more details, check the official installation guide: [https://docs.astral.sh/uv/getting-started/installation/](https://docs.astral.sh/uv/getting-started/installation/)

### Install and Run ZSim

1.  **Clone or Download:** Get the latest source code from the [release page](https://github.com/ZZZSimulator/ZSim/releases) or clone the repository using `git clone`.
2.  **Navigate:** Open your terminal in the project directory.
3.  **Sync Dependencies:**

    ```bash
    uv sync
    ```

4.  **Run the Simulator:**

    ```bash
    uv run zsim run
    ```

## Development

### Key Components

*   **Simulation Engine:** Core logic located in `zsim/simulator/`
*   **Web API:** FastAPI-based REST API in `zsim/api_src/`
*   **Web UI:** Streamlit-based interface in `zsim/webui.py` and a Vue.js + Electron desktop application in `electron-app/`
*   **CLI:** Command-line interface via `zsim/run.py`
*   **Database:** SQLite-based storage for character/enemy configurations
*   **Electron App:** Desktop application built with Vue.js and Electron, communicating with the FastAPI backend.

### Build System

The project uses a Make-based build system.

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

**Note:** The `pnpm dev` command offers the best development experience.

### Testing Structure

*   Unit tests in `tests/`
*   API tests in `tests/api/`
*   Fixtures defined in `tests/conftest.py`
*   Uses pytest with asyncio support

```bash
# Run the tests
uv run pytest
# Run the tests with coverage report
uv run pytest -v --cov=zsim --cov-report=html
```

## TODO LIST

Check the [develop guide](https://github.com/ZZZSimulator/ZSim/wiki/%E8%B4%A1%E7%8C%AE%E6%8C%87%E5%8D%97-Develop-Guide) for more information.

## Environment Variables

### FastAPI Backend

*   `ZSIM_DISABLE_ROUTES` - Disable API routes (set to "1", default: enabled)
*   `ZSIM_IPC_MODE` - IPC communication mode: "auto", "uds", or "http" (default: "auto")
*   `ZSIM_UDS_PATH` - UDS socket file path (default: "/tmp/zsim_api.sock")
*   `ZSIM_API_PORT` - API server port, set to 0 for automatic port selection (default: 0)
*   `ZSIM_API_HOST` - API server host address (default: "127.0.0.1")

### IPC Mode Behavior

*   **auto**: Uses UDS on Unix-like OS, HTTP on Windows
*   **uds**: Uses Unix Domain Socket (Unix-like only)
*   **http**: Uses HTTP/TCP (default)