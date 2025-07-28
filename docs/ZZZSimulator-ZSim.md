# ZSim: Ultimate Damage Calculator for Zenless Zone Zero

Calculate and visualize team damage in Zenless Zone Zero effortlessly with ZSim, a powerful and automatic damage calculator.  [View the project on GitHub](https://github.com/ZZZSimulator/ZSim).

![ZSim Screenshot](docs/img/zsim成图.svg)
![ZSim Logo](docs/img/横板logo成图.png)

## Key Features

*   **Automatic Damage Calculation:**  Automatically simulates team actions based on Action Priority Lists (APL) for accurate damage output prediction.
*   **User-Friendly Interface:** Easily configure agent equipment and select appropriate APLs.
*   **Visualized Results:** Generates informative charts and tables to analyze team damage, including detailed breakdowns per character.
*   **Equipment Customization:** Allows you to edit and optimize agents' equipment for maximum performance.
*   **APL Customization:**  Edit APL code to tailor team strategies.

## Installation

### Prerequisites: Install UV (if you haven't already)

ZSim relies on the `uv` package manager.  Choose your operating system below:

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

**Using pip:**

```bash
pip install uv
```

For detailed `uv` installation instructions, refer to the official documentation: [https://docs.astral.sh/uv/getting-started/installation/](https://docs.astral.sh/uv/getting-started/installation/)

### Install ZSim

1.  **Navigate to the Project Directory:** Open a terminal and change directory to the location where you cloned or downloaded the ZSim project.
2.  **Create and Activate a Virtual Environment:**

    ```bash
    uv venv
    ```

3.  **Install ZSim:**

    ```bash
    uv pip install .  # The '.' refers to the current directory
    ```

## Running ZSim

**After installation:**

```bash
zsim run
```

**Without Installation (Run Directly):**

```bash
uv run ./zsim/run.py run
```

**Alternative (Run Directly):**

```bash
uv run zsim run
```

## Contributing

For more information on contributing, see the [develop guide](https://github.com/ZZZSimulator/ZSim/wiki/%E8%B4%A1%E7%8C%AE%E6%8C%87%E5%8D%97-Develop-Guide).