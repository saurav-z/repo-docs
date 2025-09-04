# ZSim: Zenless Zone Zero (ZZZ) Battle Simulator & Damage Calculator

**Unleash the power of strategic team building with ZSim, the fully automated battle simulator and damage calculator for Zenless Zone Zero!**

[View the original repository on GitHub](https://github.com/ZZZSimulator/ZSim)

## Key Features:

*   **Automated Battle Simulation:** Automatically simulates battles based on your team composition and equipment.
*   **Damage Calculation:** Accurately calculates total damage output, considering character weapons, equipment, and action priority.
*   **User-Friendly Interface:** Easily edit agent equipment and customize action priority lists (APL).
*   **Visual Reports:** Generates detailed damage information in visual charts and tables for in-depth analysis.
*   **Flexible:** Supports various gameplay strategies and character builds.

## Installation

### Prerequisites

*   **Python:** Ensure you have Python installed on your system.
*   **UV Package Manager:**  ZSim utilizes the UV package manager for dependency management. Install it using the following commands:

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

    Or check the official installation guide: [https://docs.astral.sh/uv/getting-started/installation/](https://docs.astral.sh/uv/getting-started/installation/)

### Install and Run ZSim

1.  **Clone the Repository:** Download the latest source code from the release page or use `git clone`.
2.  **Navigate to the Project Directory:** Open your terminal and navigate to the project's root directory.
3.  **Install Dependencies and Run:**

    ```bash
    uv sync
    uv run zsim run
    ```

## Development

### Key Components
1. **Simulation Engine** - Core logic in `zsim/simulator/` handles the battle simulation
2. **Web API** - FastAPI-based REST API in `zsim/api_src/` for programmatic access
3. **Web UI** - Streamlit-based interface in `zsim/webui.py` and new Vue.js + Electron desktop application in `electron-app/`
4. **CLI** - Command-line interface via `zsim/run.py`
5. **Database** - SQLite-based storage for character/enemy configurations
6. **Electron App** - Desktop application built with Vue.js and Electron that communicates with the FastAPI backend

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
- Unit tests in `tests/` directory
- API tests in `tests/api/`
- Fixtures defined in `tests/conftest.py`
- Uses pytest with asyncio support

```bash
# Run the tests
uv run pytest
# Run the tests with coverage report
uv run pytest -v --cov=zsim --cov-report=html
```
## TODO List

Check the [develop guide](https://github.com/ZZZSimulator/ZSim/wiki/%E8%B4%A1%E7%8C%AE%E6%8C%87%E5%8D%97-Develop-Guide) for more details.

## Environment Variables

### FastAPI Backend
- `ZSIM_DISABLE_ROUTES` - Set to "1" to disable API routes (default: enabled)
- `ZSIM_IPC_MODE` - IPC communication mode: "auto", "uds", or "http" (default: "http")
- `ZSIM_UDS_PATH` - UDS socket file path when using UDS mode (default: "/tmp/zsim_api.sock")
- `ZSIM_API_PORT` - API server port, set to 0 for automatic port selection (default: 0)
- `ZSIM_API_HOST` - API server host address (default: "127.0.0.1")

### IPC Mode Behavior
- **auto**: Uses uds on Unix like OS, and http on windows
- **uds**: Uses Unix Domain Socket for local communication (Unix like only)
- **http**: Uses HTTP/TCP for communication (default mode)