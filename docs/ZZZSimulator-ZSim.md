# ZSim: Damage Calculator for Zenless Zone Zero - Optimize Your Team's Performance

**Tired of guesswork? ZSim automatically calculates and visualizes your team's damage output in Zenless Zone Zero, giving you the edge you need to dominate the battlefield.**

[View the original repository on GitHub](https://github.com/ZZZSimulator/ZSim)

## Key Features

*   **Automated Damage Calculation:**  Simulates combat scenarios and calculates total damage output based on your team's composition and equipment.
*   **Visualized Results:** Generates clear charts and tables to present detailed damage information for each character, making analysis easy.
*   **Equipment Customization:**  Easily edit your agents' equipment to fine-tune your team's build.
*   **Action Priority List (APL) Editing:** Customize the APL to simulate various gameplay strategies.
*   **User-Friendly Interface:** Intuitive interface simplifies the process of calculating and analyzing damage.

## Installation

ZSim utilizes `uv` for package management.  Follow these steps to get started:

### Install UV (if you haven't already)

Choose the appropriate installation method for your operating system:

*   **macOS or Linux:**

    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

*   **Windows 11 (24H2 or later):**

    ```bash
    winget install --id=astral-sh.uv  -e
    ```

*   **Older Windows versions:**

    ```powershell
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    ```

*   **Alternative: Using pip**

    ```bash
    pip install uv
    ```

    Or check the official installation guide: [https://docs.astral.sh/uv/getting-started/installation/](https://docs.astral.sh/uv/getting-started/installation/)

### Install ZZZ-Simulator

1.  Navigate to the project directory in your terminal.
2.  Create a virtual environment and install the package:

    ```bash
    uv venv
    uv pip install .  # The '.' refers to the relative path
    ```

## Running ZSim

After installation, you can run ZSim from any terminal:

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

For more details on planned features and how you can contribute, see the [Develop Guide](https://github.com/ZZZSimulator/ZSim/wiki/%E8%B4%A1%E7%8C%AE%E6%8C%87%E5%8D%97-Develop-Guide).