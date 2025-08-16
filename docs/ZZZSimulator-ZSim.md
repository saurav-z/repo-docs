# ZSim: The Ultimate Zenless Zone Zero Damage Calculator & Battle Simulator

**Maximize your team's potential in Zenless Zone Zero with ZSim, a powerful and user-friendly battle simulator!**  ([Original Repo](https://github.com/ZZZSimulator/ZSim))

[![ZSim Project Logo](docs/img/横板logo成图.png)](https://github.com/ZZZSimulator/ZSim)

## Key Features:

*   **Automated Battle Simulation:**  Simulates battles automatically based on Action Priority Lists (APLs), eliminating the need for manual skill sequence input.
*   **Comprehensive Damage Calculation:** Calculates total damage output for your team, taking into account character weapons, equipment, and buffs.
*   **User-Friendly Interface:** Offers an intuitive interface for easy setup and analysis.
*   **Visualized Results:** Generates clear and informative charts and tables for detailed damage reports.
*   **Character Customization:**  Allows you to edit your agents' equipment to optimize performance.
*   **APL Editing:**  Provides the ability to customize APLs for advanced strategy.
*   **Detailed Damage Breakdown:** Displays individual character damage information.

## Installation

### Prerequisites: Install `UV` Package Manager

Choose your installation method:

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

Or consult the official UV installation guide: [https://docs.astral.sh/uv/getting-started/installation/](https://docs.astral.sh/uv/getting-started/installation/)

### Install and Run ZSim

1.  **Navigate:** Open your terminal and navigate to the project's directory.
2.  **Sync Dependencies:**
    ```bash
    uv sync
    ```
3.  **Run the Simulator:**
    ```bash
    uv run zsim run
    ```

## Development

### Key Components

*   **Simulation Engine:** `zsim/simulator/` (Core battle simulation logic)
*   **Web API:** `zsim/api_src/` (FastAPI-based REST API)
*   **Web UI:** `zsim/webui.py` (Streamlit interface)
*   **Desktop App:** `electron-app/` (Vue.js + Electron application)
*   **CLI:** `zsim/run.py` (Command-line interface)
*   **Database:** SQLite (for character/enemy configurations)

### Development Setup

```bash
# Install dependencies
uv sync

# For WebUI Development
uv run zsim run

# For FastAPI Backend
uv run zsim api

# For Electron App Development (also install Node.js dependencies)
cd electron-app
yarn install
```

### Testing

*   **Unit Tests:** Located in the `tests/` directory.
*   **API Tests:** Located in the `tests/api/` directory.
*   **Fixtures:** Defined in `tests/conftest.py`.
*   **Testing Framework:** Uses pytest with asyncio support.

### Running Tests

```bash
# Run all tests
uv run pytest

# Run tests with coverage report
uv run pytest -v --cov=zsim --cov-report=html
```

## Further Development

Consult the [development guide](https://github.com/ZZZSimulator/ZSim/wiki/%E8%B4%A1%E7%8C%AE%E6%8C%87%E5%8D%97-Develop-Guide) for detailed information about contribution and future development plans.