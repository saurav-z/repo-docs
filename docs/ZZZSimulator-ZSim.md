# ZSim: Zenless Zone Zero Damage Calculator

Maximize your team's potential in Zenless Zone Zero with ZSim, a powerful and automated damage calculator. [See the original repository](https://github.com/ZZZSimulator/ZSim)

## Key Features

*   **Automated Damage Calculation:**  Calculates total team damage output without manual skill sequence input (APL-based).
*   **User-Friendly Interface:**  Easily edit agent equipment and select APLs.
*   **Detailed Reporting:** Generates visual charts and tables for in-depth damage analysis.
*   **Character-Specific Data:** Provides granular damage information for each agent.
*   **Flexible Configuration:**  Edit agent equipment and APL code to tailor simulations.

## Installation

### Prerequisites: Install `UV` (if you don't have it)

`UV` is a fast Python package installer. Install it using one of the following methods:

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

**Using `pip`:**

```bash
pip install uv
```

For detailed instructions, consult the official UV documentation: [https://docs.astral.sh/uv/getting-started/installation/](https://docs.astral.sh/uv/getting-started/installation/)

### Install ZSim

1.  **Navigate to the project directory** in your terminal (e.g., where you downloaded the source code).
2.  **Create a virtual environment and install dependencies:**

    ```bash
    uv venv
    uv pip install .  # '.' refers to the current directory
    ```

## Running ZSim

Once installed, you can run ZSim:

```bash
zsim run
```

Alternatively, you can run it directly without installation:

```bash
uv run ./zsim/run.py run
```

```bash
# or also:
uv run zsim run
```

## Future Development

Refer to the [Development Guide](https://github.com/ZZZSimulator/ZSim/wiki/%E8%B4%A1%E7%8C%AE%E6%8C%87%E5%8D%97-Develop-Guide) for information on upcoming features and how to contribute.