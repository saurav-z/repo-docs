<div align="center" style="text-align: center;">

<h1>opendbc: Your Open-Source Car Control API</h1>
<p>
  <b>Take control of your car's steering, gas, and brakes with opendbc, the powerful Python API.</b>
</p>

<h3>
  <a href="https://docs.comma.ai">Docs</a>
  <span> · </span>
  <a href="https://github.com/commaai/openpilot/blob/master/docs/CONTRIBUTING.md">Contribute</a>
  <span> · </span>
  <a href="https://discord.comma.ai">Discord</a>
</h3>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![X Follow](https://img.shields.io/twitter/follow/comma_ai)](https://x.com/comma_ai)
[![Discord](https://img.shields.io/discord/469524606043160576)](https://discord.comma.ai)

</div>

---

opendbc is a Python API that allows you to interact with and control your car's systems, including steering, gas, and brakes. Built by [comma.ai](https://comma.ai) and the open-source community, opendbc unlocks the potential of your vehicle, enabling advanced driver-assistance features and more. Get started with opendbc to explore the possibilities of automotive control.  See the [original repo](https://github.com/commaai/opendbc).

## Key Features

*   **Control:** Take command of steering, acceleration, and braking systems.
*   **Read Data:** Access vital vehicle information like speed and steering angle.
*   **Open Source:** Contribute to a growing community and shape the future of car control.
*   **ADAS Focus:** Primarily designed to support ADAS interfaces, including openpilot.
*   **Extensible:** Support for reading and writing various car data, such as EV charge status and door locks.
*   **Python API:** A developer-friendly Python API for easy integration and use.
*   **Extensive Car Support:** Compatible with many cars with LKAS and ACC.

## Quick Start

Get up and running with opendbc quickly:

```bash
git clone https://github.com/commaai/opendbc.git
cd opendbc

# Run the all-in-one script for dependency installation, building, linting, and testing:
./test.sh

# Or, run individual commands:
pip3 install -e .[testing,docs]  # Install dependencies
scons -j8                        # Build (using 8 cores)
pytest .                         # Run tests
lefthook run lint                # Run the linter
```

Explore example programs in the [`examples/`](examples/) directory for practical demonstrations of reading car state and controlling various systems, including a joystick control example.

## Project Structure

*   [`opendbc/dbc/`](opendbc/dbc/): Contains [DBC](https://en.wikipedia.org/wiki/CAN_bus#DBC_(CAN_Database_Files)) files for different cars.
*   [`opendbc/can/`](opendbc/can/): Provides a library for parsing and building CAN messages from DBC files.
*   [`opendbc/car/`](opendbc/car/): Offers a high-level Python library for car interaction.
*   [`opendbc/safety/`](opendbc/safety/): Manages the functional safety aspects for supported cars.

## How to Port a Car

Extend opendbc's capabilities by adding support for new cars.

### Steps to Port a Car

1.  **Connect to the Car:** Use a comma 3X and a car harness to connect to the vehicle's CAN buses.
2.  **Structure of a Port:** Understand that car port implementations typically reside within `opendbc/car/<brand>/`.
3.  **Reverse Engineer CAN Messages:** Use tools like [cabana](https://github.com/commaai/openpilot/tree/master/tools/cabana) to decode CAN messages.
4.  **Tuning:** Optimize control performance.

### Key Files within a car port:

*   `carstate.py`: Parses information from the CAN stream using the car's DBC file.
*   `carcontroller.py`: Outputs CAN messages to control the car.
*   `<brand>can.py`: Python helpers to build CAN messages.
*   `fingerprints.py`: Identifies car models.
*   `interface.py`: High-level class for car interaction.
*   `radar_interface.py`: Parses radar data.
*   `values.py`: Enumerates supported cars.

## Contributing

Join the opendbc community and contribute via GitHub and [Discord](https://discord.comma.ai). Engage in discussions and follow the `#dev-opendbc-cars` channel.

### Roadmap

*   **Short Term:**
    *   `pip install opendbc`
    *   100% Type and Line Coverage
    *   Improve car port development: refactors, tools, tests, and documentation
    *   Improve car state exposure
*   **Longer Term:**
    *   Expand car support for LKAS + ACC
    *   Automate lateral and longitudinal control evaluation
    *   Implement auto-tuning for lateral and longitudinal control
    *   Implement Automatic Emergency Braking

## Safety Model

The opendbc safety firmware is designed for use with [openpilot](https://github.com/commaai/openpilot) and [panda](https://github.com/commaai/panda). When the safety firmware is enabled, the CAN buses are silent by default in `SAFETY_SILENT` mode. Safety modes, such as `SAFETY_ALLOUTPUT`, are disabled by default and require custom builds. The safety modes support controls_allowed, allowing or blocking messages based on board state.

## Code Rigor

The `safety` folder is written with code rigor in mind due to its importance.

*   Code analysis via [cppcheck](https://github.com/danmar/cppcheck/) and [MISRA C:2012](https://misra.org.uk/) checks
*   Strict compiler options (`-Wall -Wextra -Wstrict-prototypes -Werror`)
*   Unit tests ([opendbc/safety/tests](opendbc/safety/tests)) for all car variants
*   Mutation testing on MISRA coverage
*   100% line coverage for unit tests

The car interface library uses the [ruff linter](https://github.com/astral-sh/ruff) and [mypy](https://mypy-lang.org/).

## Bounties

Earn bounties for contributing to opendbc!

*   $2000 - [Any car brand / platform port](https://github.com/orgs/commaai/projects/26/views/1?pane=issue&itemId=47913774)
*   $250 - [Any car model port](https://github.com/orgs/commaai/projects/26/views/1?pane=issue&itemId=47913790)
*   $300 - [Reverse Engineering a new Actuation Message](https://github.com/orgs/commaai/projects/26/views/1?pane=issue&itemId=73445563)
*   Additional bounties for popular cars available at [comma.ai/bounties](comma.ai/bounties).

## FAQ

*   **How do I use this?** A [comma 3X](https://comma.ai/shop/comma-3x) is designed for running and developing opendbc and openpilot.
*   **Which cars are supported?** Refer to the [supported cars list](docs/CARS.md).
*   **Can I add support for my car?** Yes! See the guide [here](https://github.com/commaai/opendbc/blob/docs/README.md#how-to-port-a-car).
*   **Which cars can be supported?** Cars with LKAS and ACC. More info [here](https://github.com/commaai/openpilot/blob/master/docs/CARS.md#dont-see-your-car-here).
*   **How does this work?** Hardware replaces your car's lane keep and adaptive cruise features. See [this talk](https://www.youtube.com/watch?v=FL8CxUSfipM).
*   **Is there a timeline or roadmap for adding car support?** Community driven with final validation by comma.

### Terms

*   **port**: Adding support for a specific car
*   **lateral control**: Steering control
*   **longitudinal control**: Gas/brakes control
*   **fingerprinting**: Automatic car identification
*   **[LKAS](https://en.wikipedia.org/wiki/Lane_departure_warning_system)**: Lane keeping assist
*   **[ACC](https://en.wikipedia.org/wiki/Adaptive_cruise_control)**: Adaptive cruise control
*   **[harness](https://comma.ai/shop/car-harness)**: Hardware to connect to the car
*   **[panda](https://github.com/commaai/panda)**: Hardware for CAN bus access
*   **[ECU](https://en.wikipedia.org/wiki/Electronic_control_unit)**: Car control modules
*   **[CAN bus](https://en.wikipedia.org/wiki/CAN_bus)**: Car communication bus
*   **[cabana](https://github.com/commaai/openpilot/tree/master/tools/cabana#readme)**: Tool for CAN message analysis
*   **[DBC file](https://en.wikipedia.org/wiki/CAN_bus#DBC)**: CAN message definitions
*   **[openpilot](https://github.com/commaai/openpilot)**: ADAS system using opendbc
*   **[comma](https://github.com/commaai)**: The company behind opendbc
*   **[comma 3X](https://comma.ai/shop/comma-3x)**: Hardware for running openpilot

### More Resources

*   Videos and tools from COMMA_CON for in-depth explanations and community contributions.
*   Datasets and tools for CAN data analysis and development.

## Join the Team: [comma.ai/jobs](https://comma.ai/jobs)

comma.ai is hiring engineers to work on opendbc and [openpilot](https://github.com/commaai/openpilot).