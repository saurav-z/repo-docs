# opendbc: Your Car's Python API

> Take control of your car's systems with `opendbc`, a powerful Python API for reading and writing data to your vehicle's CAN bus, unlocking advanced features and control.  [View the original repo](https://github.com/commaai/opendbc).

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![X Follow](https://img.shields.io/twitter/follow/comma_ai)](https://x.com/comma_ai)
[![Discord](https://img.shields.io/discord/469524606043160576)](https://discord.comma.ai)

</div>

---

## Key Features

*   **Control Your Car:**  Access and manipulate gas, brake, and steering systems.
*   **Real-Time Data:** Read crucial information like speed and steering angle.
*   **Wide Vehicle Support:** Designed to support a growing number of cars with LKAS and ACC features.
*   **Community Driven:**  Benefit from a vibrant community and contribute to expanding vehicle compatibility.
*   **Safety Focused:**  Built with robust safety models and rigorous code rigor for secure operation.
*   **Open Source:** Leverage open-source code for advanced driver-assistance systems (ADAS).

## Quickstart & Installation

```bash
git clone https://github.com/commaai/opendbc.git
cd opendbc

# Install dependencies, build, and run tests all in one go:
./test.sh
```

Example programs are in the [`examples/`](examples/) directory, including `examples/joystick.py` which allows you to control your car using a joystick.

## Project Structure

*   `opendbc/dbc/`: Contains [DBC files](https://en.wikipedia.org/wiki/CAN_bus#DBC) that define CAN messages.
*   `opendbc/can/`: A library for parsing and building CAN messages from DBC files.
*   `opendbc/car/`: A high-level library for interacting with cars using Python.
*   `opendbc/safety/`:  The functional safety layer for cars supported by `opendbc/car/`.

## How to Port a Car: Expand Vehicle Support

Extend `opendbc` support to your car by following these steps:

1.  **Connect to the Car:** Use a comma 3X and car harness for connection.
2.  **Understand Port Structure:**  Familiarize yourself with the car-specific folder structure (e.g., `opendbc/car/<brand>/`).
3.  **Reverse Engineer CAN Messages:** Use tools like [cabana](https://github.com/commaai/openpilot/tree/master/tools/cabana) to analyze CAN data.
4.  **Tuning:**  Use the longitudinal maneuvers report to tune your car's longitudinal control.

For a deeper dive into porting a car, read the dedicated documentation within the repo.

## Contribute & Community

*   **GitHub:**  All development is coordinated on GitHub.
*   **Discord:** Join the `#dev-opendbc-cars` channel on Discord to connect with the community.

### Roadmap

*   **Short Term:**
    *   `pip install opendbc`
    *   100% type and line coverage.
    *   Simplify car porting with better tools and documentation.
    *   Improve car support state.
*   **Longer Term:**
    *   Support all cars with LKAS + ACC.
    *   Automated tuning for lateral and longitudinal control.
    *   Implement Automated Emergency Braking.

## Safety Model

The opendbc safety firmware, used in conjunction with [openpilot](https://github.com/commaai/openpilot) and [panda](https://comma.ai/shop/panda), is written for the use within [openpilot](https://github.com/commaai/openpilot). The safety firmware provides and enforces the [openpilot safety](https://github.com/commaai/openpilot/blob/master/docs/SAFETY.md). Due to its critical function, it's important that the application code rigor within the `safety` folder is held to high standards.

## Code Rigor

The opendbc safety firmware has robust code rigor in place. The following [CI regression tests](https://github.com/commaai/opendbc/actions) are performed:

*   Static code analysis using [cppcheck](https://github.com/danmar/cppcheck/), including specific checks for [MISRA C:2012](https://misra.org.uk/) violations.
*   Strict compiler options, including flags like `-Wall -Wextra -Wstrict-prototypes -Werror`.
*   [Unit tests](opendbc/safety/tests) and [mutation tests](opendbc/safety/tests/misra/test_mutation.py) to ensure code quality.
*   [ruff linter](https://github.com/astral-sh/ruff) and [mypy](https://mypy-lang.org/) for the car interface library.

## Bounties

Contribute to opendbc and earn bounties:

*   **Any car brand / platform port:** $2000
*   **Any car model port:** $250
*   **Reverse Engineering a new Actuation Message:** $300

Also, check out [comma.ai/bounties](https://comma.ai/bounties) for higher-value bounties for popular car models.

## FAQ

*   **How do I use this?**  A [comma 3X](https://comma.ai/shop/comma-3x) is recommended for running and developing with opendbc and openpilot.
*   **Which cars are supported?** See the [supported cars list](docs/CARS.md).
*   **Can I add support for my car?** Yes!  Follow the car porting guide.
*   **Which cars can be supported?** Cars with LKAS and ACC.
*   **How does this work?**  Replaces car's lane keep and adaptive cruise features. See [this talk](https://www.youtube.com/watch?v=FL8CxUSfipM) for a more detailed explanation.
*   **Is there a timeline or roadmap for adding car support?**  Community-driven, with comma doing final validation.

## Key Terms

*   **port**: Integrating support for a specific car.
*   **lateral control**: Steering control.
*   **longitudinal control**: Gas/brakes control.
*   **fingerprinting**: Automatic car identification.
*   **LKAS**: Lane Keeping Assist.
*   **ACC**: Adaptive Cruise Control.
*   **harness**: Hardware to connect to the car's ADAS system.
*   **panda**: Hardware to access the car's CAN bus.
*   **ECU**: Electronic Control Unit, car's computer.
*   **CAN bus**: Internal car network.
*   **cabana**: CAN message reverse engineering tool.
*   **DBC file**: Defines CAN messages.
*   **openpilot**: ADAS system using opendbc.
*   **comma**: The company behind opendbc.
*   **comma 3X**: The hardware used to run openpilot.

## Additional Resources

*   [How Do We Control The Car?](https://www.youtube.com/watch?v=nNU6ipme878&pp=ygUoY29tbWEgY29uIDIwMjEgaG93IGRvIHdlIGNvbnRyb2wgdGhlIGNhcg%3D%3D)
*   [How to Port a Car](https://www.youtube.com/watch?v=XxPS5TpTUnI&t=142s&pp=ygUPamFzb24gY29tbWEgY29u)
*   [commaCarSegments](https://huggingface.co/datasets/commaai/commaCarSegments)
*   [cabana](https://github.com/commaai/openpilot/tree/master/tools/cabana#readme)
*   [can_print_changes.py](https://github.com/commaai/openpilot/blob/master/selfdrive/debug/can_print_changes.py)
*   [longitudinal maneuvers](https://github.com/commaai/openpilot/tree/master/tools/longitudinal_maneuvers)
*   [opendbc data](https://commaai.github.io/opendbc-data/)

## Join the Team!  [comma.ai/jobs](https://comma.ai/jobs)

comma is hiring engineers to work on opendbc and [openpilot](https://github.com/commaai/openpilot).