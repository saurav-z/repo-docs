# ZSim: Zenless Zone Zero Battle Simulator & Damage Calculator

**Maximize your Zenless Zone Zero team's potential with ZSim, the ultimate battle simulator and damage calculator!**  ([Original Repository](https://github.com/ZZZSimulator/ZSim))

![ZSim Logo](docs/img/横板logo成图.png)

## About ZSim

ZSim is a powerful, **fully automated** battle simulator designed specifically for Zenless Zone Zero (ZZZ). It eliminates the need for manual skill sequence input, allowing you to focus on optimizing your team's performance. Simply equip your agents, select an Action Priority List (APL), and run the simulation to analyze damage output and team effectiveness.

## Key Features

*   **Automatic Battle Simulation:** Simulates battles based on your chosen APL, eliminating manual action input.
*   **Comprehensive Damage Calculation:** Calculates total damage output for your team composition, taking into account character stats, equipment, and weapon characteristics.
*   **Visual Reporting:** Generates insightful visual charts and tables to display damage information and performance metrics.
*   **Agent Customization:**  Easily edit agent equipment to explore different build options.
*   **APL Configuration:** Edit the Action Priority List code to customize your team's behavior.
*   **User-Friendly Interface:** Provides an intuitive interface for easy simulation setup and result analysis.

## Installation

### Prerequisites

Before installing ZSim, ensure you have the `uv` package manager installed. Follow the appropriate instructions below based on your operating system:

**Using `pip` (if Python is installed):**

```bash
pip install uv
```

**macOS or Linux:**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows 11 24H2 or later:**

```bash
winget install --id=astral-sh.uv  -e
```

**Older Windows Versions:**

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

For detailed `uv` installation instructions, see the official guide: [https://docs.astral.sh/uv/getting-started/installation/](https://docs.astral.sh/uv/getting-started/installation/)

### Installing and Running ZSim

1.  **Download:** Download the latest source code from the release page or use `git clone`.

2.  **Navigate:** Open a terminal in the project directory.

3.  **Install Dependencies & Run:**

    ```bash
    uv sync
    uv run zsim run
    ```

## Development

### Core Components

*   **Simulation Engine:**  Located in `zsim/simulator/`, houses the core battle simulation logic.
*   **Web API:** A FastAPI-based REST API in `zsim/api_src/` for programmatic access.
*   **Web UI:**  Utilizes a Streamlit-based interface in `zsim/webui.py` and a new Vue.js + Electron desktop application in `electron-app/`.
*   **CLI:** Command-line interface accessible via `zsim/run.py`.
*   **Database:** SQLite database for storing character and enemy configurations.
*   **Electron App:** A desktop application built with Vue.js and Electron that communicates with the FastAPI backend.

### Development Setup

1.  **Install `uv`:** (See Installation instructions above)

2.  **Project Setup:**

    ```bash
    uv sync
    ```

3.  **Run WebUI:**
    ```bash
    uv run zsim run
    ```
4.  **Run FastAPI Backend:**
    ```bash
    uv run zsim api
    ```

5.  **Electron App Development:**
    ```bash
    cd electron-app
    yarn install
    ```

### Testing

*   **Unit Tests:** Located in the `tests/` directory.
*   **API Tests:** Located in `tests/api/`.
*   **Fixtures:** Defined in `tests/conftest.py`.
*   **Test Framework:** Utilizes pytest with asyncio support.

**Running Tests:**

```bash
uv run pytest
```

**Running Tests with Coverage Report:**

```bash
uv run pytest -v --cov=zsim --cov-report=html
```

## Further Development

For detailed information on contributing and future development, check the [development guide](https://github.com/ZZZSimulator/ZSim/wiki/%E8%B4%A1%E7%8C%AE%E6%8C%87%E5%8D%97-Develop-Guide).