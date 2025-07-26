# ZSim: Zenless Zone Zero Damage Calculator - Optimize Your Team's DPS!

[![ZSim Logo](docs/img/zsim成图.svg)](https://github.com/ZZZSimulator/ZSim)

Tired of guessing team damage in Zenless Zone Zero? ZSim is a powerful damage calculator designed to analyze your team's performance automatically, providing detailed insights into your damage output. This allows you to optimize your team compositions and perfect your strategies.

[View the original repository on GitHub](https://github.com/ZZZSimulator/ZSim)

## Key Features

*   **Automated Damage Calculation:** No manual skill sequence setup required (though support for sequence mode is available upon request).
*   **Team Composition Analysis:** Calculate total damage output based on your chosen team and equipment.
*   **Visual Reports:** Generate informative charts and tables to visualize damage data.
*   **Detailed Character Breakdown:** Get in-depth damage information for each character in your team.
*   **Customizable Equipment:** Easily edit character equipment to simulate different builds.
*   **APL Editing:** Modify Action Priority Lists (APLs) to fine-tune team behavior.

## Installation

### Prerequisites: Install UV (Universal Virtual Environment)

ZSim uses UV for managing its virtual environment. Choose your operating system below:

*   **macOS or Linux:**

    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

*   **Windows 11 (24H2 or later):**

    ```bash
    winget install --id=astral-sh.uv  -e
    ```

*   **Older Windows Versions:**

    ```powershell
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    ```

*   **Alternative (using pip):**

    ```bash
    pip install uv
    ```

    For more detailed installation instructions, refer to the official UV documentation:  [https://docs.astral.sh/uv/getting-started/installation/](https://docs.astral.sh/uv/getting-started/installation/)

### Install ZSim

1.  **Navigate to the Project Directory:** Open your terminal and change the directory to where you downloaded or cloned the ZSim project.
2.  **Create and Activate the Virtual Environment and Install Dependencies:**

    ```bash
    uv venv
    uv pip install .
    ```

## Running ZSim

1.  **From Anywhere (After Installation):**

    ```bash
    zsim run
    ```

2.  **Direct Execution (Without Installation):**

    ```bash
    uv run ./zsim/run.py run
    ```

    ```bash
    # Or, a simplified version:
    uv run zsim run
    ```

## Future Development

Explore the [Development Guide](https://github.com/ZZZSimulator/ZSim/wiki/%E8%B4%A1%E7%8C%AE%E6%8C%87%E5%8D%97-Develop-Guide) for information on upcoming features and how you can contribute.