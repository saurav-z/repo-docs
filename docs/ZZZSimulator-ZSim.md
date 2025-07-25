# ZSim: Calculate Zenless Zone Zero Damage with Ease

Tired of manual calculations? **ZSim is your all-in-one damage calculator for Zenless Zone Zero, providing automated and insightful team damage analysis.**

[View the original repository on GitHub](https://github.com/ZZZSimulator/ZSim)

![ZSim Logo](docs/img/zsim成图.svg)

![ZSim Project Group Logo](docs/img/横板logo成图.png)

## Key Features:

*   **Automated Damage Calculation:** Automatically simulates team actions based on your chosen APL (Action Priority List), eliminating the need for manual skill sequence input.
*   **Team Composition Analysis:** Calculates total damage output for your selected team composition.
*   **Visual Reports:** Generates interactive charts and tables to visualize damage data, providing detailed insights for each character.
*   **Agent Customization:** Easily edit and configure your agents' equipment for accurate simulations.
*   **APL Editing:** Customize or use predefined Action Priority Lists to fine-tune your simulations.

## Installation

### Prerequisites: Install UV (Universal Virtual Environment)

UV is a fast and efficient virtual environment and package manager.  Choose your operating system below:

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

**Alternative (using pip):**

```bash
pip install uv
```

For more detailed installation instructions for UV, visit: [https://docs.astral.sh/uv/getting-started/installation/](https://docs.astral.sh/uv/getting-started/installation/)

### Install ZSim

1.  Open your terminal and navigate to the project directory.
2.  Create a virtual environment and install ZSim:

    ```bash
    uv venv
    uv pip install .  # Install from the current directory ('.')
    ```

## Run ZSim

After installation, you can run ZSim from any directory:

```bash
zsim run
```

**Alternative Run (without installation):**

You can also run ZSim directly using `uv`:

```bash
uv run ./zsim/run.py run
```

```bash
# or also:
uv run zsim run
```

## Further Development

For information on contributing and future development, please refer to the [development guide](https://github.com/ZZZSimulator/ZSim/wiki/%E8%B4%A1%E7%8C%AE%E6%8C%87%E5%8D%97-Develop-Guide).