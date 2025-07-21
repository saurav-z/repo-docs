# ZSim: The Ultimate Damage Calculator for Zenless Zone Zero

Unleash the power of your team and optimize your damage output with ZSim, a user-friendly damage calculator for Zenless Zone Zero, now available on [GitHub](https://github.com/ZZZSimulator/ZSim).

![ZSim Logo](docs/img/横板logo成图.png)

## Key Features

*   **Automatic Damage Simulation:**  ZSim automatically simulates battles based on your chosen Action Priority List (APL), eliminating the need for manual skill sequence input.
*   **Team Composition Analysis:** Calculate total damage output for your entire team composition, taking into account character, weapon, and equipment stats.
*   **Visual Reports:** Generate insightful charts and tables to visualize your team's damage performance.
*   **Detailed Damage Breakdown:**  View comprehensive damage information for each character in your team.
*   **Customizable Builds:** Easily edit your agents' equipment and tailor them to your playstyle.
*   **APL Customization:**  Modify the APL code to fine-tune your simulations.

## Installation

### Prerequisites

*   **UV (Universal Virtual Environment):**  A fast and modern Python package and virtual environment manager. Install it using one of the following methods:

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
    *   **Pip (Alternative):**
        ```bash
        pip install uv
        ```
    *   For a complete guide, consult the official [UV Installation Guide](https://docs.astral.sh/uv/getting-started/installation/).

### ZSim Installation

1.  **Clone or Download:** Obtain the ZSim source code by either cloning the repository or downloading the latest release from the [releases page](https://github.com/ZZZSimulator/ZSim/releases).
2.  **Navigate:** Open your terminal in the project directory.
3.  **Create and Activate Virtual Environment & Install Dependencies:**
    ```bash
    uv venv
    uv pip install .  # Install the ZSim package from the current directory
    ```

## Running ZSim

1.  **Run from Anywhere (Recommended):**
    ```bash
    zsim run
    ```
2.  **Run Directly (Alternative - If ZSim not installed):**
    ```bash
    uv run ./zsim/run.py run
    # or
    uv run zsim run
    ```

##  Contributing

For information about contributing to the project, please refer to the [Development Guide](https://github.com/ZZZSimulator/ZSim/wiki/%E8%B4%A1%E7%8C%AE%E6%8C%87%E5%8D%97-Develop-Guide).