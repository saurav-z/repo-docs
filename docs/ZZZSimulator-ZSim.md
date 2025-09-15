# ZSim: Zenless Zone Zero Battle Simulator & Damage Calculator

**Maximize your team's potential in Zenless Zone Zero with ZSim, a powerful and user-friendly battle simulator.** [View the original repository](https://github.com/ZZZSimulator/ZSim)

## Key Features:

*   **Automated Battle Simulation:** Eliminates the need for manual skill sequence input, streamlining the simulation process.
*   **Detailed Damage Calculation:** Accurately assesses total damage output, considering character equipment and weapon stats.
*   **Visual Data Representation:** Generates interactive charts and tables for insightful analysis of your team's performance.
*   **Agent Equipment Customization:** Allows you to modify and optimize agent equipment to find the best builds.
*   **APL (Action Priority List) Editing:** Customize your team's action sequence to experiment and refine your strategy.
*   **User-Friendly Interface:** Offers an intuitive platform for easy team composition and analysis.

## Installation

### Prerequisites

1.  **UV Package Manager:** Install UV using the following command in your terminal:

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
        winget install --id=astral-sh.uv  -e
        ```
    *   **On lower versions of Windows:**
        ```bash
        powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
        ```
    *   For more information, consult the official UV installation guide: [https://docs.astral.sh/uv/getting-started/installation/](https://docs.astral.sh/uv/getting-started/installation/)

### Installing and Running ZSim

1.  **Navigate to the project directory** in your terminal.
2.  **Sync Dependencies:**
    ```bash
    uv sync
    ```
3.  **Run the Simulator:**
    ```bash
    uv run zsim run
    ```

## Development

### Project Structure

*   **Simulation Engine:** `zsim/simulator/` houses the core battle simulation logic.
*   **Web API:** `zsim/api_src/` features a FastAPI-based REST API.
*   **Web UI:** Streamlit-based interface in `zsim/webui.py` and Vue.js + Electron desktop application in `electron-app/`.
*   **CLI:** Command-line interface available via `zsim/run.py`.
*   **Database:** SQLite-based database for character and enemy configurations.
*   **Electron App:** Desktop application built with Vue.js and Electron that interfaces with the FastAPI backend.

### Build System

The project leverages a Make-based build system for managing development, build, and release tasks.

#### Available Make Targets:

*   **Build:** `make build` (Full build), `make backend` (Backend API only), `make electron-build` (Electron app only).
*   **Development:** `make dev` (Frontend development server), `make clean`, `make check`.
*   **Utilities:** `make help`.

### Setup and Running

1.  **Install Dependencies:**

    ```bash
    uv sync
    ```
2.  **Development Servers:**

    *   **WebUI:**
        ```bash
        uv run zsim run
        ```
    *   **FastAPI Backend:**
        ```bash
        uv run zsim api
        ```
    *   **Electron App Development (requires Node.js dependencies):**
        ```bash
        cd electron-app
        pnpm install
        pnpm dev  # Recommended: one-click frontend & backend development server.
        ```
    *   **Electron App Production Build:**
        ```bash
        cd electron-app
        pnpm build
        ```

### Testing

*   **Testing Structure:** Unit tests in `tests/`, API tests in `tests/api/`, and fixtures in `tests/conftest.py`. Uses pytest with asyncio support.
*   **Run Tests:**
    ```bash
    uv run pytest
    ```
    ```bash
    uv run pytest -v --cov=zsim --cov-report=html # With coverage report
    ```

## Environment Variables

### FastAPI Backend

*   `ZSIM_DISABLE_ROUTES`: Disable API routes ("1" to disable). Default: enabled.
*   `ZSIM_IPC_MODE`: Inter-Process Communication mode ("auto", "uds", or "http"). Default: "auto".
*   `ZSIM_UDS_PATH`: UDS socket file path (when using UDS mode). Default: "/tmp/zsim_api.sock".
*   `ZSIM_API_PORT`: API server port (set to 0 for automatic port selection). Default: 0.
*   `ZSIM_API_HOST`: API server host address. Default: "127.0.0.1".

### IPC Mode Behavior

*   **auto:** Uses UDS on Unix-like OS and HTTP on Windows.
*   **uds:** Uses Unix Domain Socket (Unix-like only).
*   **http:** Uses HTTP/TCP for communication (default mode).

## TODO LIST

Refer to the [develop guide](https://github.com/ZZZSimulator/ZSim/wiki/%E8%B4%A1%E7%8C%AE%E6%8C%87%E5%8D%97-Develop-Guide) for further details.