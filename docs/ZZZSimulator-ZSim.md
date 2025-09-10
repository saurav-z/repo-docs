# ZSim: Zenless Zone Zero Battle Simulator & Damage Calculator 

**Optimize your team's performance with ZSim, a powerful and user-friendly battle simulator for the action-packed game Zenless Zone Zero.** ([Original Repo](https://github.com/ZZZSimulator/ZSim))

## Key Features

*   **Automatic Battle Simulation:** No need to manually input skill sequences, ZSim automates combat based on preset Action Priority Lists (APLs).
*   **Team Damage Calculation:** Accurately calculates total damage output, taking into account character weapons and equipment.
*   **Visual Reporting:** Generates easy-to-understand charts and tables to analyze battle results.
*   **Character & Equipment Customization:** Allows you to edit agents' equipment to experiment with different builds.
*   **APL Customization:** Edit APLs to tailor the simulation to your specific team strategy.
*   **User-Friendly Interface:**  Provides a streamlined interface for easy use and analysis.

## Installation

### Prerequisites

1.  **Install UV (Universal Virtual environment)** - This is required for managing project dependencies.  Install it using one of the following methods:

    *   **Using pip (if Python is installed):**

        ```bash
        pip install uv
        ```

    *   **macOS or Linux:**

        ```bash
        curl -LsSf https://astral.sh/uv/install.sh | sh
        ```

    *   **Windows 11 (24H2 or later):**

        ```bash
        winget install --id=astral-sh.uv -e
        ```

    *   **Older Windows versions:**

        ```powershell
        powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
        ```

    *   For more detailed instructions, see the official UV installation guide: [https://docs.astral.sh/uv/getting-started/installation/](https://docs.astral.sh/uv/getting-started/installation/)

### ZSim Installation and Run

1.  **Clone the repository or download the source code.**
2.  **Navigate to the project directory** in your terminal.
3.  **Install dependencies and run the application**:

    ```bash
    uv sync
    uv run zsim run
    ```

## Development

### Key Components

*   **Simulation Engine:** Core logic for battle simulation (`zsim/simulator/`).
*   **Web API:** REST API built with FastAPI (`zsim/api_src/`).
*   **Web UI:** Streamlit-based interface (`zsim/webui.py`) and a desktop application built with Vue.js and Electron (`electron-app/`).
*   **CLI:** Command-line interface (`zsim/run.py`).
*   **Database:** SQLite for character and enemy configurations.
*   **Electron App:** Desktop application built with Vue.js and Electron interacting with the FastAPI backend.

### Build System

The project uses a Make-based build system for managing development, building, and release processes.

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

**Note:** `pnpm dev` is the most convenient for development, automatically starting the Vue.js frontend and FastAPI backend, with hot reload for the frontend, and full debugging.

### Testing Structure

*   Unit tests in `tests/` directory.
*   API tests in `tests/api/`.
*   Fixtures defined in `tests/conftest.py`.
*   Uses pytest with asyncio support.

```bash
# Run the tests
uv run pytest
# Run the tests with coverage report
uv run pytest -v --cov=zsim --cov-report=html
```

## TODO List

Check the [develop guide](https://github.com/ZZZSimulator/ZSim/wiki/%E8%B4%A1%E7%8C%AE%E6%8C%87%E5%8D%97-Develop-Guide) for details.

## Environment Variables

### FastAPI Backend

*   `ZSIM_DISABLE_ROUTES` - Disable API routes (default: enabled, set to "1" to disable).
*   `ZSIM_IPC_MODE` - IPC communication mode: "auto", "uds", or "http" (default: "auto").
*   `ZSIM_UDS_PATH` - UDS socket file path when using UDS mode (default: "/tmp/zsim_api.sock").
*   `ZSIM_API_PORT` - API server port, set to 0 for automatic port selection (default: 0).
*   `ZSIM_API_HOST` - API server host address (default: "127.0.0.1").

### IPC Mode Behavior

*   **auto:** Uses UDS on Unix-like OS and HTTP on Windows.
*   **uds:** Uses Unix Domain Socket (Unix-like only).
*   **http:** Uses HTTP/TCP (default).