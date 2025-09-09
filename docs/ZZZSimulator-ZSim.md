# ZSim: Zenless Zone Zero Battle Simulator & Damage Calculator 

**Maximize your team's potential in Zenless Zone Zero with ZSim, a powerful, automated battle simulator and damage calculator.** ([Original Repository](https://github.com/ZZZSimulator/ZSim))

## Key Features:

*   **Automated Simulation:**  Simulates battles based on your team composition and chosen Action Priority List (APL), eliminating the need for manual skill sequence input.
*   **Comprehensive Damage Calculation:** Accurately calculates total damage output, considering character equipment and weapon characteristics.
*   **Visual Reporting:** Generates detailed damage information, presented in user-friendly visual charts and tables.
*   **Agent Customization:** Allows you to edit and optimize your agents' equipment directly within the simulator.
*   **APL Editing:** Customize and edit APL (Action Priority List) codes to tailor your team's strategy.
*   **FastAPI Backend:** Supports programmatic access to the simulation via a REST API.
*   **Streamlit & Electron UI:**  Offers both a user-friendly Streamlit web interface and a new Vue.js + Electron desktop application for easy access.

## Getting Started

### Installation

1.  **Install UV (Package Manager):**

    *   **Using pip (if you have Python installed):**

        ```bash
        pip install uv
        ```

    *   **On macOS or Linux:**

        ```bash
        curl -LsSf https://astral.sh/uv/install.sh | sh
        ```

    *   **On Windows 11 24H2 or later:**

        ```bash
        winget install --id=astral-sh.uv -e
        ```

    *   **On lower versions of Windows:**

        ```bash
        powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
        ```
    *   Or check the official installation guide: [https://docs.astral.sh/uv/getting-started/installation/](https://docs.astral.sh/uv/getting-started/installation/)

2.  **Install and Run ZSim:**

    Open terminal in the project directory:
    ```bash
    uv sync
    uv run zsim run
    ```

## Development

### Key Components:

*   **Simulation Engine:** Core logic within `zsim/simulator/` handles battle simulations.
*   **Web API:** FastAPI-based REST API located in `zsim/api_src/`.
*   **Web UI:** Streamlit-based interface in `zsim/webui.py`.
*   **Desktop App:** New Vue.js + Electron desktop application in `electron-app/`.
*   **CLI:** Command-line interface accessible via `zsim/run.py`.
*   **Database:** SQLite-based storage for character and enemy configurations.
*   **Electron App:** Desktop application built with Vue.js and Electron communicates with the FastAPI backend.

### Build System:

The project utilizes a Make-based build system for streamlined development, building, and release processes.

#### Available Make Targets:

```bash
make build           # Full build (clean + backend + electron)
make backend         # Build backend API only
make electron-build  # Build Electron desktop application only
make dev             # Start frontend development server
make clean           # Clean all build files
make check           # Check dependencies
make help            # Display help information
```

### Setup and Installation:

```bash
uv sync
# For WebUI develop
uv run zsim run
# For FastAPI backend
uv run zsim api
# For Electron App development, also install Node.js dependencies
cd electron-app
pnpm install
```

### Running the Application:

#### Quick Start (Recommended):
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
# Electron Desktop App (production build)
cd electron-app
pnpm build
```

**Note:** The `pnpm dev` command offers the most convenient development experience.

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

## Environment Variables

### FastAPI Backend:

*   `ZSIM_DISABLE_ROUTES`: Disable API routes (default: enabled)
*   `ZSIM_IPC_MODE`: IPC communication mode: "auto", "uds", or "http" (default: "auto")
*   `ZSIM_UDS_PATH`: UDS socket file path (default: "/tmp/zsim_api.sock")
*   `ZSIM_API_PORT`: API server port, set to 0 for automatic port selection (default: 0)
*   `ZSIM_API_HOST`: API server host address (default: "127.0.0.1")

### IPC Mode Behavior:

*   **auto:** Uses `uds` on Unix-like OS, `http` on Windows.
*   **uds:** Uses Unix Domain Socket (Unix-like only).
*   **http:** Uses HTTP/TCP (default).

---