# ZSim: The Ultimate Zenless Zone Zero (ZZZ) Battle Simulator

**Tired of guesswork in Zenless Zone Zero? ZSim automates battle simulations and damage calculations, giving you the edge you need!**  Check out the original repository: [https://github.com/ZZZSimulator/ZSim](https://github.com/ZZZSimulator/ZSim)

![ZSim Logo](docs/img/横板logo成图.png)  (Replace with actual image path if needed)

## Key Features

*   **Automated Battle Simulations:** Automatically simulates team actions based on preset Action Priority Lists (APLs).
*   **Comprehensive Damage Calculation:** Calculates total damage output, considering character equipment and weapons.
*   **Visual Reporting:** Generates interactive charts and tables to analyze damage data.
*   **Character & Equipment Customization:**  Allows you to edit agents' equipment and APL code.
*   **User-Friendly Interface:**  Provides an accessible interface for analyzing team compositions.

## Installation

### Prerequisites: Install UV (if you haven't already)

This project uses the UV package manager. Install it by following the instructions for your operating system.

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

### Install and Run ZSim

1.  Open a terminal in the project directory.
2.  Run:

    ```bash
    uv sync
    uv run zsim run
    ```

## Development

ZSim leverages a modular architecture with a robust build system.

### Key Components

*   **Simulation Engine:** Core logic in `zsim/simulator/`
*   **Web API:** FastAPI-based REST API in `zsim/api_src/`
*   **Web UI:** Streamlit-based interface in `zsim/webui.py` and an Electron app.
*   **CLI:** Command-line interface in `zsim/run.py`
*   **Database:** SQLite for character/enemy configurations.
*   **Electron App:** Desktop application built with Vue.js and Electron.

### Build System (Make Targets)

The project uses a Make-based build system. Use `make help` for detailed information.

#### Common Build Commands:

```bash
make build         # Full build
make backend       # Build backend API
make electron-build  # Build Electron desktop application
make dev           # Start frontend development server (recommended)
```

### Setup and Development

1.  Install UV (as described in the Installation section).
2.  For WebUI development:  `uv run zsim run`
3.  For FastAPI backend: `uv run zsim api`
4.  For Electron App Development:
    ```bash
    cd electron-app
    pnpm install
    ```

### Running the Application

#### Quick Start (Recommended)

```bash
cd electron-app
pnpm dev
```

#### Individual Components

```bash
uv run zsim run    # Streamlit WebUI
uv run zsim api    # FastAPI Backend
# For Electron desktop build:
cd electron-app
pnpm build
```

### Testing

ZSim uses pytest with asyncio support.

```bash
# Run tests
uv run pytest

# Run tests with coverage report
uv run pytest -v --cov=zsim --cov-report=html
```

## Environment Variables

### FastAPI Backend

*   `ZSIM_DISABLE_ROUTES` (default: enabled)
*   `ZSIM_IPC_MODE` (default: "auto") - "auto", "uds", or "http"
*   `ZSIM_UDS_PATH` (default: "/tmp/zsim_api.sock")
*   `ZSIM_API_PORT` (default: 0)
*   `ZSIM_API_HOST` (default: "127.0.0.1")

### IPC Mode Behavior
- **auto**: Uses uds on Unix like OS, and http on windows
- **uds**: Uses Unix Domain Socket for local communication (Unix like only)
- **http**: Uses HTTP/TCP for communication (default mode)

## TODO

See the [develop guide](https://github.com/ZZZSimulator/ZSim/wiki/%E8%B4%A1%E7%8C%AE%E6%8C%87%E5%8D%97-Develop-Guide) for the current development roadmap.