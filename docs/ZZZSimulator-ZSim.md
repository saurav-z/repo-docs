# ZSim: The Ultimate Zenless Zone Zero Damage Calculator

**Maximize your team's damage output in Zenless Zone Zero with ZSim, the automatic damage calculator.**  [Visit the original repo](https://github.com/ZZZSimulator/ZSim) for the latest updates and to contribute!

![zsim](docs/img/zsim成图.svg)

![zsim项目组](docs/img/横板logo成图.png)

## Key Features:

*   **Automated Damage Calculation:**  Automatically simulates team actions based on Action Priority Lists (APLs) to calculate total damage.
*   **User-Friendly Interface:**  Easily edit agent equipment and select APLs for quick simulations.
*   **Visual Results:**  Generates insightful charts and tables to visualize damage breakdowns.
*   **Detailed Damage Information:** Provides comprehensive damage data for each character in your team.
*   **Customizable Agents:** Allows you to edit agent equipment to optimize your team's performance.
*   **APL Code Editing:**  Customize and fine-tune APLs for advanced simulation control.

## Installation

### Prerequisites

Before installing ZSim, you'll need to install `uv`, a fast Python package manager.  Follow the instructions below based on your operating system.

### Install UV

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

**Alternative: Using pip:**

```bash
pip install uv
```

For more detailed information on installing `uv`, please refer to the official documentation: [https://docs.astral.sh/uv/getting-started/installation/](https://docs.astral.sh/uv/getting-started/installation/)

### Install ZSim

1.  **Navigate to the project directory:** Open your terminal and navigate to the directory where you downloaded or cloned the ZSim project.
2.  **Create a virtual environment:**

```bash
uv venv
```

3.  **Install ZSim:**

```bash
uv pip install .  # The '.' refers to the current project directory.
```

## Running ZSim

You can run ZSim from anywhere after installation:

```bash
zsim run
```

Alternatively, you can run ZSim directly without installing it:

```bash
uv run ./zsim/run.py run
```

```bash
# Or also:
uv run zsim run
```

## Further Development

See the [develop guide](https://github.com/ZZZSimulator/ZSim/wiki/%E8%B4%A1%E7%8C%AE%E6%8C%87%E5%8D%97-Develop-Guide) for information on contributing to the project.