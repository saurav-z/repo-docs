# ZSim: Calculate Your Zenless Zone Zero Team Damage!

Tired of guessing team damage in Zenless Zone Zero? ZSim is a powerful, **automatic damage calculator** that analyzes your team's performance. Check out the [original ZSim repository](https://github.com/ZZZSimulator/ZSim) for more details.

![ZSim Logo](docs/img/zsim成图.svg)
![ZSim Project Group Logo](docs/img/横板logo成图.png)

## Key Features

*   **Automated Damage Calculation:**  No manual skill sequence input is required (APL based).
*   **Team Composition Analysis:** Calculate total damage output based on your team's characters and equipment.
*   **Visual Reports:** Generate charts and tables for easy understanding of damage distribution.
*   **Detailed Character Information:**  View individual damage breakdowns for each agent.
*   **Equipment & APL Customization:**  Easily edit character equipment and APL (Action Priority List) code.

## Installation

### Prerequisites: Install UV (if you haven't already)

`UV` is used to create virtual env, you can use it to run `zsim`.

Choose your installation method based on your operating system:

**macOS or Linux:**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows 11 24H2 or later:**

```bash
winget install --id=astral-sh.uv  -e
```

**Older Windows versions:**

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Using pip:**

```bash
pip install uv
```

For more detailed instructions, consult the official UV installation guide: [https://docs.astral.sh/uv/getting-started/installation/](https://docs.astral.sh/uv/getting-started/installation/)

### Install ZSim

1.  Open a terminal in the project's directory.
2.  Create a virtual environment and install the ZSim package:

    ```bash
    uv venv
    uv pip install .
    ```

## Running ZSim

1.  Open a terminal from any location.
2.  Run the damage calculator using the command:

    ```bash
    zsim run
    ```

    or if you didn't install the package:

    ```bash
    uv run ./zsim/run.py run
    ```
    or

    ```bash
    uv run zsim run
    ```

## Development

See the [Develop Guide](https://github.com/ZZZSimulator/ZSim/wiki/%E8%B4%A1%E7%8C%AE%E6%8C%87%E5%8D%97-Develop-Guide) for details on contributing.