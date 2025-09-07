# ZSim: Zenless Zone Zero Battle Simulator & Damage Calculator

**Maximize your team's potential in Zenless Zone Zero with ZSim, the automated battle simulator and damage calculator.**  Explore the official repository for detailed instructions and future updates: [https://github.com/ZZZSimulator/ZSim](https://github.com/ZZZSimulator/ZSim)

## Key Features

*   **Automated Battle Simulation:** No manual skill sequence input required – ZSim handles the actions based on your chosen Action Priority List (APL).
*   **Comprehensive Damage Calculation:** Calculates total damage output for your team composition, considering character stats, weapon/equipment, and buffs.
*   **Visual Reporting:** Generates interactive charts and tables to visualize and analyze your team's performance.
*   **Agent Equipment Editor:** Allows you to customize each agent's gear and stats.
*   **APL Customization:**  Edit and optimize your team's APL for tailored simulations.
*   **User-Friendly Interface:** Provides an intuitive interface for easy simulation setup and result interpretation.

## Getting Started

### Installation

1.  **Install UV Package Manager (if you haven't already):**

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

2.  **Install and Run ZSim:**

    ```bash
    # Navigate to the project directory in your terminal
    uv sync
    uv run zsim run
    ```

## Development

### Key Components

*   **Simulation Engine:** Core battle simulation logic (`zsim/simulator/`).
*   **Web API:**  FastAPI-based REST API (`zsim/api_src/`).
*   **Web UI:** Streamlit-based interface (`zsim/webui.py`) and new Vue.js + Electron desktop application (`electron-app/`).
*   **CLI:** Command-line interface (`zsim/run.py`).
*   **Database:** SQLite-based storage.
*   **Electron App:** Desktop application (Vue.js/Electron) communicating with the FastAPI backend.

### Build System

The project utilizes a Make-based build system.

#### Available Make Targets (Examples)

```bash
# Build all components:
make build
# Build the Electron desktop application:
make electron-build
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
# One-click development server (frontend and backend):
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

**Note:** The `pnpm dev` command offers the best development experience, providing hot reload, backend output forwarding, and full debugging capabilities.

### Testing

*   **Unit Tests:** In the `tests/` directory.
*   **API Tests:** In the `tests/api/` directory.
*   **Fixtures:** Defined in `tests/conftest.py`.
*   Uses pytest with asyncio support.

```bash
# Run the tests:
uv run pytest
# Run tests with coverage report:
uv run pytest -v --cov=zsim --cov-report=html
```

## Environment Variables

### FastAPI Backend

*   `ZSIM_DISABLE_ROUTES`: Disables API routes (default: enabled).
*   `ZSIM_IPC_MODE`: IPC communication mode ("auto", "uds", or "http", default: "auto").
*   `ZSIM_UDS_PATH`: UDS socket file path (default: "/tmp/zsim_api.sock").
*   `ZSIM_API_PORT`: API server port, set to 0 for automatic port selection (default: 0).
*   `ZSIM_API_HOST`: API server host address (default: "127.0.0.1").

### IPC Mode Behavior

*   **auto**: Uses uds on Unix-like OS, and http on windows.
*   **uds**: Uses Unix Domain Socket (Unix-like OS only).
*   **http**: Uses HTTP/TCP (default).