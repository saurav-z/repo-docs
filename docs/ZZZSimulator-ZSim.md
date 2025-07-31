# ZSim: Zenless Zone Zero Battle Simulator & Damage Calculator

**Maximize your Zenless Zone Zero team's potential with ZSim, a powerful and automated battle simulator and damage calculator!**  (See the [original repository](https://github.com/ZZZSimulator/ZSim))

![ZSim Logo](docs/img/横板logo成图.png)

## Key Features

*   **Automated Battle Simulation:** Automatically simulates battles based on your team and equipment, eliminating the need for manual skill sequence input.
*   **Comprehensive Damage Calculation:** Accurately calculates total damage output, considering character stats, weapon characteristics, and equipment effects.
*   **User-Friendly Interface:** Easily edit agent equipment and select action priority lists (APLs).
*   **Visual Reports:** Generates clear, visual charts and tables to analyze battle results and damage distribution.
*   **Detailed Damage Breakdown:** Provides in-depth damage information for each character in your team.
*   **Customizable APLs:**  Allows you to edit APL code to fine-tune your team's actions.

## Installation

### Prerequisites

1.  **Install UV (Universal Virtual Environment):**  ZSim utilizes UV for dependency management. Choose one of the following installation methods:

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
    *   For more detailed information about UV installation, check the official guide: [https://docs.astral.sh/uv/getting-started/installation/](https://docs.astral.sh/uv/getting-started/installation/)

### ZSim Installation and Running

1.  **Navigate to the project directory** in your terminal.
2.  **Install dependencies:**

    ```bash
    uv sync
    ```

3.  **Run the simulator:**

    ```bash
    uv run zsim run
    ```

## Development

### Key Components

*   **Simulation Engine:** The core logic for battle simulation, located in `zsim/simulator/`.
*   **Web API:** A FastAPI-based REST API for programmatic access, found in `zsim/api_src/`.
*   **Web UI:**  A Streamlit-based interface in `zsim/webui.py` and a new Vue.js + Electron desktop application in `electron-app/`.
*   **CLI:** Command-line interface via `zsim/run.py`.
*   **Database:** SQLite-based storage for character and enemy configurations.
*   **Electron App:** A desktop application built with Vue.js and Electron that interacts with the FastAPI backend.

### Development Setup

1.  **Install UV (see instructions above).**
2.  **For Web UI development:**

    ```bash
    uv run zsim run
    ```
3.  **For FastAPI backend development:**

    ```bash
    uv run zsim api
    ```
4.  **For Electron App development, also install Node.js dependencies:**

    ```bash
    cd electron-app
    corepack install
    pnpm install
    ```

### Testing

*   **Unit Tests:** Located in the `tests/` directory.
*   **API Tests:** Located in the `tests/api/` directory.
*   **Fixtures:** Defined in `tests/conftest.py`.
*   **Test Framework:** Uses pytest with asyncio support.

**Run tests:**

```bash
uv run pytest
```

**Run tests with coverage report:**

```bash
uv run pytest -v --cov=zsim --cov-report=html
```

## Contributing

See the [develop guide](https://github.com/ZZZSimulator/ZSim/wiki/%E8%B4%A1%E7%8C%AE%E6%8C%87%E5%8D%97-Develop-Guide) for details on how to contribute to the project.