# ZZZ Simulator: Optimize Your Zenless Zone Zero Team's Damage

**Maximize your team's damage potential in Zenless Zone Zero with ZZZ Simulator, a powerful and automated damage calculator.**  Learn how to use ZSim to analyze and optimize your team compositions for maximum effectiveness. ([See the original repository](https://github.com/ZZZSimulator/ZSim))

## Key Features of ZZZ Simulator

*   **Automated Damage Calculation:** Automatically simulates team actions based on a defined Action Priority List (APL), eliminating manual skill sequence input (unless sequence mode is desired).
*   **Comprehensive Team Analysis:** Calculates total damage output for a team composition, considering individual character equipment and weapon characteristics.
*   **User-Friendly Interface:** Simplifies the process with an intuitive interface for easy configuration and analysis.
*   **Visual Reporting:** Generates visual charts and tables to present detailed damage information for each character and team performance.
*   **Equipment & APL Customization:** Allows you to edit agents' equipment and customize the APL to fine-tune your simulations.

## Installation Guide

### Prerequisites: Install UV (Universal Virtual Environment)

Follow these steps to install UV, a fast virtual environment and package manager:

**macOS or Linux:**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows 11 (24H2 or later):**

```bash
winget install --id=astral-sh.uv -e
```

**Older Windows Versions:**

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Alternatively, using pip:**

```bash
pip install uv
```

For a comprehensive installation guide, see the official documentation: [https://docs.astral.sh/uv/getting-started/installation/](https://docs.astral.sh/uv/getting-started/installation/)

### Install ZZZ Simulator

1.  **Navigate to the project directory:** Open your terminal and change directory to the location where you downloaded or cloned the ZZZ Simulator project.
2.  **Create and Activate a Virtual Environment:**

    ```bash
    uv venv
    ```
3.  **Install ZZZ Simulator:**

    ```bash
    uv pip install .
    ```

## Running ZZZ Simulator

**To run ZZZ Simulator:**

```bash
zsim run
```

**Alternatively, run it directly without installation:**

```bash
uv run ./zsim/run.py run
```

```bash
# or also:
uv run zsim run
```

## Future Development

Consult the [development guide](https://github.com/ZZZSimulator/ZSim/wiki/%E8%B4%A1%E7%8C%AE%E6%8C%87%E5%8D%97-Develop-Guide) for detailed information on future features and contributing to the project.