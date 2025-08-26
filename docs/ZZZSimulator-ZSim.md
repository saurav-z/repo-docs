# ZSim: Zenless Zone Zero Battle Simulator and Damage Calculator

**Maximize your Zenless Zone Zero team's potential with ZSim, the ultimate damage calculator and battle simulator!**  ([Original Repository](https://github.com/ZZZSimulator/ZSim))

![ZSim Project Logo](docs/img/横板logo成图.png)

## Key Features:

*   **Automated Battle Simulation:**  Automatically simulates battles based on your team's Action Priority List (APL) without manual skill sequence input.
*   **Comprehensive Damage Calculation:**  Calculates total damage output, factoring in character weapons, equipment, and buffs.
*   **Visual Data Reporting:**  Generates intuitive charts and tables to visualize damage results and character performance.
*   **Character Equipment Customization:**  Easily edit and manage your agents' equipment to optimize builds.
*   **APL Customization:**  Edit the Action Priority List (APL) code to test out different team strategies.

## Installation:

1.  **Install UV (if you haven't already):**  UV is a package manager. Choose your operating system below.  If you already have uv installed, skip this step.

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
    *   **Older Windows Versions:**
        ```powershell
        powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
        ```
    *   **For the official UV install guide:** [https://docs.astral.sh/uv/getting-started/installation/](https://docs.astral.sh/uv/getting-started/installation/)
2.  **Install and Run ZSim:**

    *   Open your terminal in the ZSim project directory.
    *   Run the following commands:
        ```bash
        uv sync
        uv run zsim run
        ```

## Development:

### Key Components:

*   **Simulation Engine:** `zsim/simulator/` (Core battle simulation logic)
*   **Web API:**  FastAPI-based REST API (`zsim/api_src/`)
*   **Web UI:** Streamlit-based interface (`zsim/webui.py`) and new Vue.js + Electron desktop application (`electron-app/`)
*   **CLI:** Command-line interface (`zsim/run.py`)
*   **Database:** SQLite-based for character and enemy data.
*   **Electron App:** Desktop application (Vue.js and Electron) that interfaces with the FastAPI backend.

### Setup and Installation (Development):

1.  **Install the UV package manager:**
    ```bash
    uv sync
    ```
2.  **WebUI Development:**
    ```bash
    uv run zsim run
    ```
3.  **FastAPI Backend Development:**
    ```bash
    uv run zsim api
    ```
4.  **Electron App Development:**
    ```bash
    cd electron-app
    yarn install
    ```

### Testing Structure:

*   **Unit Tests:**  `tests/` directory
*   **API Tests:**  `tests/api/`
*   **Fixtures:**  Defined in `tests/conftest.py`
*   **Testing Framework:** pytest with asyncio support

    ```bash
    # Run tests
    uv run pytest
    # Run tests with coverage report
    uv run pytest -v --cov=zsim --cov-report=html
    ```

## Further Information:

*   For more development details, refer to the [Develop Guide](https://github.com/ZZZSimulator/ZSim/wiki/%E8%B4%A1%E7%8C%AE%E6%8C%87%E5%8D%97-Develop-Guide).