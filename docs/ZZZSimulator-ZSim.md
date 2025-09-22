# ZSim: Zenless Zone Zero Battle Simulator & Damage Calculator

**Maximize your team's potential in Zenless Zone Zero with ZSim, a powerful and user-friendly battle simulator.**  [View the original repository on GitHub](https://github.com/ZZZSimulator/ZSim)

## Key Features:

*   **Automated Simulation:**  Simulates battles automatically based on your team's equipment and a selected Action Priority List (APL), eliminating the need for manual skill sequence input.
*   **Comprehensive Damage Calculation:** Calculates total damage output, taking into account each character's weapons, equipment, and abilities.
*   **Visual Reports:** Generates interactive charts and tables to visualize damage output and analyze battle performance.
*   **Agent Equipment Customization:**  Allows detailed editing of agents' equipment for accurate simulation.
*   **APL Editing:**  Provides the ability to customize APLs for advanced team strategies.
*   **User-Friendly Interface:**  Offers a Streamlit-based web UI and an Electron desktop app for easy interaction.

## Installation

### Prerequisites: Install UV (if you haven't already)

ZSim utilizes the UV package manager for dependency management. Follow these instructions to install UV:

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

### Install and Run ZSim

1.  **Clone the Repository:** Download the source code using `git clone` or from the releases page.

2.  **Navigate to the Project Directory:** Open your terminal in the project directory.

3.  **Install Dependencies and Run:**

    ```bash
    uv sync
    uv run zsim run
    ```

## Development

### Key Components:

*   **Simulation Engine:** Core logic for battle simulation located in `zsim/simulator/`.
*   **Web API:**  FastAPI-based REST API in `zsim/api_src/`.
*   **Web UI:**  Streamlit interface (`zsim/webui.py`) and Electron desktop application (Vue.js + Electron in `electron-app/`).
*   **CLI:**  Command-line interface via `zsim/run.py`.
*   **Database:** SQLite-based database for storing character and enemy configurations.
*   **Electron App:** Desktop application built with Vue.js and Electron, communicating with the FastAPI backend.

### Build System:

ZSim employs a comprehensive Make-based build system to streamline development, building, and release processes.

#### Available Make Targets:

*   `make build`: Full build (clean + backend + electron)
*   `make backend`: Build backend API only
*   `make electron-build`: Build Electron desktop application only
*   `make dev`: Start frontend development server
*   `make clean`: Clean all build files
*   `make check`: Check dependencies
*   `make help`: Display help information

### Setup and Installation for Development:

```bash
# Install UV package manager (if not already installed)
uv sync

# For WebUI development
uv run zsim run

# For FastAPI backend
uv run zsim api

# For Electron App development, install Node.js dependencies and then run the development server
cd electron-app
pnpm install
pnpm dev # Starts the Vue.js frontend and FastAPI backend.
```

### Running the Application:

#### Recommended (One-Click Development):

```bash
cd electron-app
pnpm dev
```

#### Individual Components:

```bash
# Streamlit WebUI
uv run zsim run

# FastAPI Backend
uv run zsim api

# Electron Desktop App (Production Build)
cd electron-app
pnpm build
```

**Note:** The `pnpm dev` command provides the most convenient development experience, including automatic frontend and backend startup, hot reloading, and debugging capabilities.

### Testing Structure:

*   Unit tests in the `tests/` directory.
*   API tests in the `tests/api/` directory.
*   Fixtures defined in `tests/conftest.py`.
*   Uses pytest with asyncio support.

```bash
# Run the tests
uv run pytest
# Run the tests with coverage report
uv run pytest -v --cov=zsim --cov-report=html
```

## TODO List

Refer to the [Develop Guide](https://github.com/ZZZSimulator/ZSim/wiki/%E8%B4%A1%E7%8C%AE%E6%8C%87%E5%8D%97-Develop-Guide) for more details.

## Environment Variables

### FastAPI Backend:

*   `ZSIM_DISABLE_ROUTES`:  Set to "1" to disable API routes (default: enabled).
*   `ZSIM_IPC_MODE`: IPC communication mode: "auto", "uds", or "http" (default: "auto").
*   `ZSIM_UDS_PATH`: UDS socket file path when using UDS mode (default: "/tmp/zsim_api.sock").
*   `ZSIM_API_PORT`: API server port, set to 0 for automatic port selection (default: 0).
*   `ZSIM_API_HOST`: API server host address (default: "127.0.0.1").

### IPC Mode Behavior:

*   **auto:** Uses UDS on Unix-like OS and HTTP on Windows.
*   **uds:** Uses Unix Domain Socket for local communication (Unix-like only).
*   **http:** Uses HTTP/TCP for communication (default mode).