# opendbc: A Python API for Your Car - Control and Understand Your Vehicle's Data

**opendbc** empowers you to take control of your car's systems, providing a Python API to read and write data, enabling advanced features like steering, gas, and brake control.  Dive into the world of automotive data and build the ultimate vehicle management app!  Learn more and contribute at the [opendbc GitHub repository](https://github.com/commaai/opendbc).

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![X Follow](https://img.shields.io/twitter/follow/comma_ai)](https://x.com/comma_ai)
[![Discord](https://img.shields.io/discord/469524606043160576)](https://discord.comma.ai)

## Key Features

*   **Control & Monitor:** Control steering, gas, brakes, and more, while reading essential data like speed and steering angle.
*   **Open Source:**  Leverage the power of open source to customize and enhance your car's capabilities.
*   **Extensive Car Support:**  Effortlessly interface with a wide range of vehicles, especially those with LKAS and ACC.
*   **Easy to Use:**  Utilize the Python API to create custom applications for your car.
*   **Community Driven:**  Benefit from a vibrant community that constantly expands vehicle support and features.

## Quick Start

Get up and running with opendbc quickly:

```bash
git clone https://github.com/commaai/opendbc.git
cd opendbc
./test.sh # Installs dependencies, builds, and runs tests
```

Explore example programs in the [`examples/`](examples/) directory.  For example, [`examples/joystick.py`](examples/joystick.py) lets you control a car with a joystick.

## Project Structure

*   [`opendbc/dbc/`](opendbc/dbc/): Stores DBC files for CAN message definitions.
*   [`opendbc/can/`](opendbc/can/): Provides a library for parsing and constructing CAN messages.
*   [`opendbc/car/`](opendbc/car/): Offers a high-level Python library for interacting with cars.
*   [`opendbc/safety/`](opendbc/safety/): Contains functional safety implementations for supported vehicles.

## How to Port a Car

Extend opendbc to your specific vehicle by following these steps:

1.  **Connect:** Connect to the car using a comma 3X and a car harness.
2.  **Reverse Engineer CAN Messages:** Use tools like [cabana](https://github.com/commaai/openpilot/tree/master/tools/cabana) to analyze CAN data.
3.  **Develop a Car Port:**  Create the following files within `opendbc/car/<brand>/`:

    *   `carstate.py`: Parses CAN data.
    *   `carcontroller.py`: Outputs CAN messages for control.
    *   `<brand>can.py`: Provides helper functions for building CAN messages.
    *   `fingerprints.py`: Stores ECU firmware versions.
    *   `interface.py`: Defines the high-level car interface.
    *   `radar_interface.py`: Parses radar data (if applicable).
    *   `values.py`: Lists supported car models.
4.  **Tuning:** Use the [longitudinal maneuvers](https://github.com/commaai/openpilot/tree/master/tools/longitudinal_maneuvers) report to evaluate and tune your car's control.

## Contributing

All opendbc development takes place on GitHub and Discord. Visit the `#dev-opendbc-cars` channel and `Vehicle Specific` section.

## Roadmap

*   **Short Term:**
    *   `pip install opendbc`
    *   Achieve 100% type and line coverage.
    *   Improve car port creation with refactors, tools, tests, and documentation.
    *   Better state reporting.
*   **Longer Term:**
    *   Expand support to every car with LKAS + ACC.
    *   Implement automatic lateral and longitudinal control/tuning evaluation.
    *   Develop auto-tuning for lateral and longitudinal control.
    *   Incorporate Automatic Emergency Braking.

## Safety Model

When used with [opendbc safety firmware](opendbc/safety) and a [panda](https://comma.ai/shop/panda), the system defaults to `SAFETY_SILENT` mode, which ensures CAN bus silence.  Select a safety mode to send messages.

The safety firmware, used with [openpilot](https://github.com/commaai/openpilot), provides critical safety.

## Code Rigor

The `safety` folder is held to high standards. We have the following in place:

*   Static code analysis with [cppcheck](https://github.com/danmar/cppcheck/).
*   [MISRA C:2012](https://misra.org.uk/) violations check.
*   Strict compiler flags `-Wall -Wextra -Wstrict-prototypes -Werror`.
*   Unit tests for each car variant with [mutation tests](opendbc/safety/tests/misra/test_mutation.py).
*   100% line coverage enforced.
*   [ruff linter](https://github.com/astral-sh/ruff) and [mypy](https://mypy-lang.org/) on the car interface library.

## Bounties

Earn rewards for your contributions:

*   $2000 - [Any car brand / platform port](https://github.com/orgs/commaai/projects/26/views/1?pane=issue&itemId=47913774)
*   $250 - [Any car model port](https://github.com/orgs/commaai/projects/26/views/1?pane=issue&itemId=47913790)
*   $300 - [Reverse Engineering a new Actuation Message](https://github.com/orgs/commaai/projects/26/views/1?pane=issue&itemId=73445563)

Additional bounties are offered for popular cars. See [comma.ai/bounties](comma.ai/bounties).

## FAQ

*   **How do I use this?** A [comma 3X](https://comma.ai/shop/comma-3x) is recommended.
*   **Which cars are supported?** See the [supported cars list](docs/CARS.md).
*   **Can I add support for my car?** Yes, read the [guide](https://github.com/commaai/opendbc/blob/docs/README.md#how-to-port-a-car).
*   **Which cars can be supported?** Any car with LKAS and ACC (more info [here](https://github.com/commaai/openpilot/blob/master/docs/CARS.md#dont-see-your-car-here)).
*   **How does this work?** It replaces built-in lane keep and adaptive cruise features. See [this talk](https://www.youtube.com/watch?v=FL8CxUSfipM) for details.
*   **Is there a timeline or roadmap for car support?** No set timeline; car support is community-driven with comma validation.

### Terms

*   **port**:  Specific car integration.
*   **lateral control**: Steering control.
*   **longitudinal control**: Gas/brakes control.
*   **fingerprinting**: Automatic car identification.
*   **[LKAS](https://en.wikipedia.org/wiki/Lane_departure_warning_system)**: Lane Keeping Assist.
*   **[ACC](https://en.wikipedia.org/wiki/Adaptive_cruise_control)**: Adaptive Cruise Control.
*   **[harness](https://comma.ai/shop/car-harness)**: Car-specific hardware.
*   **[panda](https://github.com/commaai/panda)**: Hardware for CAN bus access.
*   **[ECU](https://en.wikipedia.org/wiki/Electronic_control_unit)**: Car computers.
*   **[CAN bus](https://en.wikipedia.org/wiki/CAN_bus)**: Car communication bus.
*   **[cabana](https://github.com/commaai/openpilot/tree/master/tools/cabana#readme)**: CAN message reverse engineering tool.
*   **[DBC file](https://en.wikipedia.org/wiki/CAN_bus#DBC)**: CAN message definitions.
*   **[openpilot](https://github.com/commaai/openpilot)**: ADAS system.
*   **[comma](https://github.com/commaai)**: The company behind opendbc.
*   **[comma 3X](https://comma.ai/shop/comma-3x)**: Hardware for running openpilot.

### More resources

*   [*How Do We Control The Car?*](https://www.youtube.com/watch?v=nNU6ipme878&pp=ygUoY29tbWEgY29uIDIwMjEgaG93IGRvIHdlIGNvbnRyb2wgdGhlIGNhcg%3D%3D)
*   [*How to Port a Car*](https://www.youtube.com/watch?v=XxPS5TpTUnI&t=142s&pp=ygUPamFzb24gY29tbWEgY29u)
*   [commaCarSegments](https://huggingface.co/datasets/commaai/commaCarSegments)
*   [cabana](https://github.com/commaai/openpilot/tree/master/tools/cabana#readme)
*   [can_print_changes.py](https://github.com/commaai/openpilot/blob/master/selfdrive/debug/can_print_changes.py)
*   [longitudinal maneuvers](https://github.com/commaai/openpilot/tree/master/tools/longitudinal_maneuvers)
*   [opendbc data](https://commaai.github.io/opendbc-data/)

## Come Work with Us - [comma.ai/jobs](https://comma.ai/jobs)

Join the comma team and contribute to opendbc and [openpilot](https://github.com/commaai/openpilot). We love hiring contributors!