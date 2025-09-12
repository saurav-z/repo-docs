# ZSim: Zenless Zone Zero Battle Simulator & Damage Calculator

**Maximize your Zenless Zone Zero team's potential with ZSim, the automated battle simulator that provides in-depth damage analysis and team optimization!**

[View the ZSim Repository on GitHub](https://github.com/ZZZSimulator/ZSim)

## Key Features

*   **Automated Battle Simulation:** Automatically simulates battles based on your team's equipment and chosen Action Priority List (APL).
*   **Detailed Damage Calculation:** Calculates total damage output, considering character weapons, equipment, and buffs.
*   **Visual Reports:** Generates easy-to-understand charts and tables to analyze damage data.
*   **Team Customization:** Edit agent equipment and customize APLs to fine-tune your strategies.
*   **User-Friendly Interface:** Features a Streamlit-based web UI and a desktop application for ease of use.

## Installation

### Prerequisites

*   **Python:** Ensure you have Python installed on your system.
*   **UV Package Manager:** Install UV, a fast Python package manager. Follow the instructions below based on your operating system:

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

    *   **Older Windows versions:**
        ```powershell
        powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
        ```

### Install and Run ZSim

1.  **Clone the Repository:** If you haven't already, download the source code by either cloning the repository or downloading the latest release from the releases page.
2.  **Navigate to Project Directory:** Open a terminal in the project directory.
3.  **Sync Dependencies:**
    ```bash
    uv sync
    ```
4.  **Run the Application:**
    ```bash
    uv run zsim run
    ```

## Development

ZSim utilizes a modular architecture with key components:

*   **Simulation Engine:** Located in `zsim/simulator/`.
*   **Web API:** A FastAPI-based REST API in `zsim/api_src/`.
*   **Web UI:** Uses Streamlit in `zsim/webui.py` and a Vue.js + Electron desktop application in `electron-app/`.
*   **CLI:** Command-line interface via `zsim/run.py`.
*   **Database:** An SQLite-based database for configuration.
*   **Electron App:** Desktop app built with Vue.js and Electron.

### Build System

The project employs a Make-based build system for efficient development and deployment.

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

*   Unit tests are located in the `tests/` directory.
*   API tests can be found in `tests/api/`.
*   Test fixtures are defined in `tests/conftest.py`.
*   The project uses pytest with asyncio support for testing.

```bash
# Run the tests
uv run pytest
# Run the tests with coverage report
uv run pytest -v --cov=zsim --cov-report=html
```

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