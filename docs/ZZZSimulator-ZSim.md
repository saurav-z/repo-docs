# ZSim: The Ultimate Zenless Zone Zero Damage Calculator

**Maximize your team's DPS with ZSim, a powerful and automated damage calculator for Zenless Zone Zero.**

![ZSim Logo](./docs/img/zsim成图.svg)

![ZSim Project Team Logo](./docs/img/横板logo成图.png)

## Key Features:

*   **Automated Damage Calculation:**  Simulates team actions based on Action Priority Lists (APLs) without manual skill sequence input, providing accurate total damage output.
*   **Team Composition Analysis:**  Calculates total damage based on your team's composition, taking into account character equipment and weapons.
*   **Visual Reporting:** Generates easy-to-understand visual charts and tables to analyze damage output and character performance.
*   **Detailed Damage Breakdown:** Provides in-depth damage information for each character in your team.
*   **Equipment & APL Editing:** Allows you to customize character equipment and APLs for precise simulations.

## Installation

### Prerequisites:

Before installing ZSim, ensure you have `uv` installed.  Follow the appropriate instructions for your operating system:

**Install UV:**

```bash
# macOS or Linux:
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```bash
# Windows 11 24H2 or later:
winget install --id=astral-sh.uv  -e
```

```bash
# Lower versions of Windows:
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

```bash
# Using pip:
pip install uv
```

For more detailed `uv` installation instructions, see the official documentation: [https://docs.astral.sh/uv/getting-started/installation/](https://docs.astral.sh/uv/getting-started/installation/)

### Installing ZSim:

1.  **Navigate to Project Directory:** Open your terminal and navigate to the directory where you downloaded the ZSim project.
2.  **Create a Virtual Environment:**

    ```bash
    uv venv
    ```
3.  **Install ZSim:**

    ```bash
    uv pip install .  # The '.' refers to the current directory
    ```

## Running ZSim

Once installed, run the damage calculator from your terminal:

```bash
zsim run
```

Alternatively, you can run it directly using `uv`:

```bash
uv run ./zsim/run.py run
```

```bash
# or also:
uv run zsim run
```

## Contribute & Learn More

For detailed development information and contribution guidelines, see the [Development Guide](https://github.com/ZZZSimulator/ZSim/wiki/%E8%B4%A1%E7%8C%AE%E6%8C%87%E5%8D%97-Develop-Guide).

**Visit the GitHub Repository:**  [https://github.com/ZZZSimulator/ZSim](https://github.com/ZZZSimulator/ZSim)