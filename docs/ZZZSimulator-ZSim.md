# ZSim: The Ultimate Zenless Zone Zero Damage Calculator

Tired of guessing team damage in Zenless Zone Zero?  ZSim provides a powerful, automated damage calculator to help you optimize your team compositions and maximize your combat effectiveness. [Check out the original ZSim repository on GitHub!](https://github.com/ZZZSimulator/ZSim)

## Key Features

*   **Automated Damage Calculation:**  ZSim automatically simulates team actions based on Action Priority Lists (APLs), eliminating the need for manual skill sequence input (unless you request it!).
*   **Detailed Damage Analysis:** Provides in-depth damage breakdowns for each character within your team, allowing for granular optimization.
*   **Team Composition Optimization:**  Calculates total team damage output, considering character equipment, weapon stats, and buffs.
*   **User-Friendly Interface:** Easily edit agent equipment and customize APLs to simulate various scenarios.
*   **Visualized Results:** Generates easy-to-understand visual charts and tables to present damage reports.
*   **APL Customization:** Customize the Action Priority Lists (APLs) to simulate different combat strategies.

## Installation

### Prerequisites: Install UV (Universal Virtualenv)

ZSim leverages UV for environment management. If you don't have UV installed, follow the instructions below for your operating system.

**macOS or Linux:**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows 11 (24H2 or later):**

```bash
winget install --id=astral-sh.uv  -e
```

**Older Windows Versions:**

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Alternative: Install UV with pip:**

```bash
pip install uv
```

For detailed UV installation instructions, see: [https://docs.astral.sh/uv/getting-started/installation/](https://docs.astral.sh/uv/getting-started/installation/)

### Install ZSim

1.  **Get the Source Code:** Download the latest release from the releases page or clone the repository using `git clone`.
2.  **Navigate to the Project Directory:** Open a terminal and navigate to the directory where you downloaded/cloned ZSim.
3.  **Create a Virtual Environment and Install:**

    ```bash
    uv venv
    uv pip install .  # The '.' refers to the current directory.
    ```

## Running ZSim

After installation, you can run ZSim from any directory.

```bash
zsim run
```

**Alternative: Run Without Installation:**

If you prefer not to install ZSim globally, you can run it directly using UV:

```bash
uv run ./zsim/run.py run
```

```bash
# or also:
uv run zsim run
```

## Contributing

For information on contributing, please see the [development guide](https://github.com/ZZZSimulator/ZSim/wiki/%E8%B4%A1%E7%8C%AE%E6%8C%87%E5%8D%97-Develop-Guide).