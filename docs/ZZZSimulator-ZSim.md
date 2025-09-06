# ZSim: Zenless Zone Zero Battle Simulator & Damage Calculator

**Maximize your team's damage output in Zenless Zone Zero with ZSim, a powerful, automated battle simulator and damage calculator!**  Check out the [original repository](https://github.com/ZZZSimulator/ZSim) for the source code.

## Key Features

*   **Automated Battle Simulation:** Automatically simulates battles based on your team composition and chosen action priority lists (APLs).
*   **Precise Damage Calculation:** Calculate total damage output, considering character equipment and weapon characteristics.
*   **User-Friendly Interface:** Easily edit agent equipment, select APLs, and run simulations through a simple interface.
*   **Visual Results:** Generate comprehensive reports in visual charts and tables for detailed analysis.
*   **Detailed Character Breakdown:** Provides damage information for each character, allowing for in-depth analysis.
*   **Flexible APL Editing:** Customize and modify APL code to fine-tune your team's performance.

## Installation

### Prerequisites

*   **Python:** Ensure you have Python installed on your system.
*   **UV Package Manager:** This project uses the [UV package manager](https://docs.astral.sh/uv/getting-started/installation/) for managing dependencies.  Install it using the following commands:

    ```bash
    # Using pip if you have python installed:
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

### Installation Steps

1.  **Clone the Repository:** Download the source code using `git clone` or download it from the [Releases](https://github.com/ZZZSimulator/ZSim/releases).
2.  **Navigate to the Project Directory:** Open a terminal or command prompt and navigate to the project directory.
3.  **Install Dependencies:**  Run `uv sync` in the project directory.
4.  **Run the ZSim Application:**

    ```bash
    uv run zsim run
    ```

    Or, for the FastAPI backend and Electron App:
    ```bash
    # For WebUI develop
    uv run zsim run 
    # For FastAPI backend
    uv run zsim api

    # For Electron App development, also install Node.js dependencies
    cd electron-app
    pnpm install
    ```

    For the Electron App (production build):
    ```bash
    cd electron-app
    pnpm build
    ```

## Development

### Key Components

*   **Simulation Engine:** Core logic in `zsim/simulator/`
*   **Web API:** FastAPI-based REST API in `zsim/api_src/`
*   **Web UI:** Streamlit-based interface in `zsim/webui.py` and new Vue.js + Electron desktop application in `electron-app/`
*   **CLI:** Command-line interface via `zsim/run.py`
*   **Database:** SQLite-based storage
*   **Electron App:** Desktop application built with Vue.js and Electron

### Build System

The project uses a Make-based build system for managing the development, building, and release processes.

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

## Environment Variables

### FastAPI Backend

*   `ZSIM_DISABLE_ROUTES` - Set to "1" to disable API routes (default: enabled)
*   `ZSIM_IPC_MODE` - IPC communication mode: "auto", "uds", or "http" (default: "auto")
*   `ZSIM_UDS_PATH` - UDS socket file path when using UDS mode (default: "/tmp/zsim_api.sock")
*   `ZSIM_API_PORT` - API server port, set to 0 for automatic port selection (default: 0)
*   `ZSIM_API_HOST` - API server host address (default: "127.0.0.1")

### IPC Mode Behavior

*   **auto**: Uses uds on Unix-like OS and http on Windows.
*   **uds**: Uses Unix Domain Socket for local communication (Unix-like only).
*   **http**: Uses HTTP/TCP for communication (default mode).