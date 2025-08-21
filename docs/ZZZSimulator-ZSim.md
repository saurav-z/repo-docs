# ZSim: Zenless Zone Zero Battle Simulator and Damage Calculator

**Maximize your Zenless Zone Zero team's potential with ZSim, the automated battle simulator that calculates damage output and analyzes team compositions.**  [View the original repository on GitHub](https://github.com/ZZZSimulator/ZSim).

![ZSim Logo](./docs/img/横板logo成图.png)

## Key Features

*   **Automated Simulation:**  Automatically simulates battles based on your chosen Action Priority List (APL), eliminating the need for manual skill sequence input.
*   **Comprehensive Damage Calculation:**  Calculates total damage output, considering agent equipment and weapon characteristics.
*   **Visual Reports:** Generates insightful charts and tables for easy analysis of team performance.
*   **Detailed Damage Breakdown:** Provides in-depth damage information for each character.
*   **Flexible Customization:**  Allows you to edit agent equipment and APL codes.
*   **User-Friendly Interface:**  Offers an intuitive interface for easy team setup and analysis.

## Installation

### Prerequisites

1.  **UV Package Manager:**  ZSim uses the UV package manager for dependency management.  Install UV using one of the following methods:

    *   **Using pip (if Python is installed):**

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

    *   For more detailed installation instructions for UV, see the official documentation: [https://docs.astral.sh/uv/getting-started/installation/](https://docs.astral.sh/uv/getting-started/installation/)

### Installing and Running ZSim

1.  **Download:** Download the latest source code from the [releases page](link to releases page, if available) or clone the repository using `git clone`.
2.  **Navigate:** Open a terminal in the project's directory.
3.  **Install Dependencies:**
    ```bash
    uv sync
    ```
4.  **Run the Simulator:**
    ```bash
    uv run zsim run
    ```

## Development

### Key Components

*   **Simulation Engine:**  The core logic for battle simulation is located in the `zsim/simulator/` directory.
*   **Web API:**  A FastAPI-based REST API is available in `zsim/api_src/` for programmatic access.
*   **Web UI:**  The user interface is built with Streamlit (`zsim/webui.py`) and a Vue.js + Electron desktop application (`electron-app/`).
*   **CLI:**  The command-line interface is provided via `zsim/run.py`.
*   **Database:** Character and enemy configurations are stored in an SQLite database.
*   **Electron App:** A desktop application developed with Vue.js and Electron that interacts with the FastAPI backend.

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
yarn install
```

### Testing Structure

*   **Unit Tests:** Found in the `tests/` directory.
*   **API Tests:** Located in `tests/api/`.
*   **Fixtures:** Defined in `tests/conftest.py`.
*   Uses pytest with asyncio support.

```bash
# Run the tests
uv run pytest
# Run the tests with coverage report
uv run pytest -v --cov=zsim --cov-report=html
```

## TODO List

See the [Development Guide](https://github.com/ZZZSimulator/ZSim/wiki/%E8%B4%A1%E7%8C%AE%E6%8C%87%E5%8D%97-Develop-Guide) for more details.