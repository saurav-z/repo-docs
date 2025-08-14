# ZSim: Zenless Zone Zero Battle Simulator & Damage Calculator

**Unleash the power of your Zenless Zone Zero team with ZSim, an automated battle simulator and damage calculator that takes the guesswork out of team building!**

[Link to Original Repo: https://github.com/ZZZSimulator/ZSim](https://github.com/ZZZSimulator/ZSim)

## Key Features

*   **Automated Battle Simulation:**  Simulates battles based on your team's equipment and an Action Priority List (APL), eliminating manual skill sequence input.
*   **Comprehensive Damage Calculation:**  Calculates total damage output, considering character weapons, equipment, and team composition.
*   **User-Friendly Interface:** Offers an intuitive interface to easily edit agent equipment and APL codes.
*   **Visual Reports:** Generates insightful reports with visual charts and detailed damage information for each character.
*   **Flexible APL Editing:** Customize team actions with APL code to optimize performance.

## Installation

### Prerequisites: UV Package Manager

ZSim requires the UV package manager.  Install it using one of the following methods:

```bash
# Using pip (if you have Python installed)
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
# On older Windows versions:
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

For more detailed UV installation instructions, see the official guide: [https://docs.astral.sh/uv/getting-started/installation/](https://docs.astral.sh/uv/getting-started/installation/)

### Installing and Running ZSim

1.  **Clone or Download:** Download the latest source code from the [releases page](https://github.com/ZZZSimulator/ZSim/releases) or use `git clone`.
2.  **Navigate:** Open a terminal in the project's directory.
3.  **Sync Dependencies:**
    ```bash
    uv sync
    ```
4.  **Run the Simulator:**
    ```bash
    uv run zsim run
    ```

## Development

ZSim is built with a modular architecture, including these key components:

*   **Simulation Engine:** Core battle simulation logic in `zsim/simulator/`.
*   **Web API:**  FastAPI-based REST API in `zsim/api_src/`.
*   **Web UI:** Streamlit-based interface (`zsim/webui.py`) and a new Vue.js + Electron desktop application (`electron-app/`).
*   **CLI:** Command-line interface (`zsim/run.py`).
*   **Database:** SQLite database for character and enemy configurations.
*   **Electron App:** Desktop application (Vue.js and Electron) interacting with the FastAPI backend.

### Setting up for Development

1.  **Install UV:** Follow the installation instructions above.
2.  **Sync Dependencies:**
    ```bash
    uv sync
    ```
3.  **Run WebUI (for development):**
    ```bash
    uv run zsim run
    ```
4.  **Run FastAPI Backend (for development):**
    ```bash
    uv run zsim api
    ```
5.  **Electron App Development:**
    ```bash
    cd electron-app
    yarn install
    ```

### Testing

ZSim uses pytest for testing.

*   **Unit Tests:** Located in the `tests/` directory.
*   **API Tests:** Located in the `tests/api/` directory.
*   **Fixtures:** Defined in `tests/conftest.py`.
*   **Run Tests:**
    ```bash
    uv run pytest
    ```
    ```bash
    uv run pytest -v --cov=zsim --cov-report=html
    ```

## Further Information

For details on contributing, please consult the [Develop Guide](https://github.com/ZZZSimulator/ZSim/wiki/%E8%B4%A1%E7%8C%AE%E6%8C%87%E5%8D%97-Develop-Guide).