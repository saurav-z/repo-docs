<div align="center">
  <h1>opendbc: Your Python API for Advanced Vehicle Control</h1>
  <p><b>Unlock the power of your car with opendbc, a Python API that allows you to control and read data from your vehicle's systems.</b></p>

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

opendbc is a powerful Python API designed to interact with your car's advanced driver-assistance systems (ADAS).  It enables you to control critical functions like steering, gas, and brakes, as well as read real-time data such as speed and steering angle.  This project supports a wide range of vehicles with the goal of providing comprehensive control over your car's systems.

## Key Features

*   **Control & Read:** Access and manipulate your car's steering, gas, and braking systems. Read crucial data like speed and steering angle.
*   **Broad Vehicle Support:** Designed to support a wide variety of vehicles with LKAS (Lane Keeping Assist System) and ACC (Adaptive Cruise Control) capabilities.
*   **ADAS Integration:** Core support for ADAS interfaces, specifically tailored for use with [openpilot](https://github.com/commaai/openpilot).
*   **Expandable Functionality:** Read and write various data points beyond basic control, including EV charge status and door locking.
*   **Open Source and Community Driven:** Contribute to a project actively developed on GitHub and supported by a thriving community on Discord.

## Getting Started

Follow these steps to get started with opendbc:

```bash
git clone https://github.com/commaai/opendbc.git
cd opendbc

# The easiest way to get started. Installs dependencies, compiles, lints, and runs tests.
./test.sh

# Individual Commands
pip3 install -e .[testing,docs]  # Install dependencies
scons -j8                        # Build with 8 cores
pytest .                         # Run the tests
lefthook run lint                # Run the linter
```

Explore the `examples/` directory for sample programs to read car state and control steering, gas, and brakes.  `examples/joystick.py` provides a convenient way to control your car using a joystick.

## Project Structure

*   [`opendbc/dbc/`](opendbc/dbc/) : Contains [DBC](https://en.wikipedia.org/wiki/CAN_bus#DBC_(CAN_Database_Files)) files, which are crucial for understanding and interpreting CAN bus data.
*   [`opendbc/can/`](opendbc/can/) :  A library for parsing and building CAN messages using the DBC files.
*   [`opendbc/car/`](opendbc/car/) : The high-level Python library for interfacing with various car models.
*   [`opendbc/safety/`](opendbc/safety/) : Implements functional safety features for the cars supported in `opendbc/car/`.

## How to Port a Car

Extend opendbc to your car by following these steps:

### Connect to the Car

1.  Connect to the car using a [comma 3X](https://comma.ai/shop/comma-3x) and a compatible car harness. If a harness is not available, start with a "developer harness".
2.  Connect to the two different CAN buses and split one of those buses to send your own actuation messages.

### Structure of a Port

A car port in `opendbc/car/<brand>/` includes:

*   `carstate.py`: Parses data from the CAN stream using the car's DBC file.
*   `carcontroller.py`: Sends CAN messages to control the car.
*   `<brand>can.py`: Provides Python helpers to build CAN messages.
*   `fingerprints.py`: Stores ECU firmware versions to identify car models.
*   `interface.py`: A high-level class for interfacing with the car.
*   `radar_interface.py`: Parses radar data.
*   `values.py`: Enumerates supported car models for the brand.

### Reverse Engineer CAN Messages

1.  Record a route with interesting events (LKAS, ACC, steering).
2.  Analyze the recording using [cabana](https://github.com/commaai/openpilot/tree/master/tools/cabana) to understand the CAN messages.

### Tuning

Evaluate and tune longitudinal control using the [longitudinal maneuvers](https://github.com/commaai/openpilot/tree/master/tools/longitudinal_maneuvers) report.

## Contributing

Contribute to opendbc on [GitHub](https://github.com/commaai/opendbc) and connect with the community on [Discord](https://discord.comma.ai).  See the `#dev-opendbc-cars` channel and `Vehicle Specific` section.

### Roadmap

*   **Short Term:**
    *   `pip install opendbc`
    *   100% type and line coverage.
    *   Refactor for easier car port development, tools, tests, and documentation.
    *   Improve the state exposure of all supported cars.
*   **Longer Term:**
    *   Expand support to all cars with LKAS and ACC interfaces.
    *   Automated lateral and longitudinal control evaluation.
    *   Auto-tuning for lateral and longitudinal control.
    *   Implement Automatic Emergency Braking.

## Safety Model

The opendbc safety firmware enforces safety standards, particularly when used with [openpilot](https://github.com/commaai/openpilot) and [panda](https://comma.ai/shop/panda). It operates in `SAFETY_SILENT` mode by default, requiring a safety mode selection to send messages.

## Code Rigor

The `safety` folder prioritizes code rigor with:

*   Static code analysis via [cppcheck](https://github.com/danmar/cppcheck/), including [MISRA C:2012](https://misra.org.uk/) compliance.
*   Strict compiler options: `-Wall -Wextra -Wstrict-prototypes -Werror`.
*   Unit tests for each supported car variant to verify the safety logic.
*   Mutation tests on the MISRA coverage and 100% line coverage for the safety unit tests.
*   Use of the [ruff linter](https://github.com/astral-sh/ruff) and [mypy](https://mypy-lang.org/) on the car interface library.

## Bounties

Contribute to opendbc and earn bounties:

*   $2000 - [Any car brand / platform port](https://github.com/orgs/commaai/projects/26/views/1?pane=issue&itemId=47913774)
*   $250 - [Any car model port](https://github.com/orgs/commaai/projects/26/views/1?pane=issue&itemId=47913790)
*   $300 - [Reverse Engineering a new Actuation Message](https://github.com/orgs/commaai/projects/26/views/1?pane=issue&itemId=73445563)

Higher value bounties available for more popular cars at [comma.ai/bounties](comma.ai/bounties).

## Frequently Asked Questions

*   **How do I use this?** Requires a [comma 3X](https://comma.ai/shop/comma-3x) hardware.
*   **Which cars are supported?** See the [supported cars list](docs/CARS.md).
*   **Can I add support for my car?** Yes! Follow the guide in the [README](https://github.com/commaai/opendbc/blob/docs/README.md#how-to-port-a-car).
*   **Which cars can be supported?** Cars with LKAS and ACC can be supported. More info [here](https://github.com/commaai/openpilot/blob/master/docs/CARS.md#dont-see-your-car-here).
*   **How does this work?** Hardware replaces lane keep and adaptive cruise features. See [this talk](https://www.youtube.com/watch?v=FL8CxUSfipM).
*   **Is there a timeline for adding car support?** Community-driven, with comma doing final safety validation.

### Definitions

*   **port**: Support for a specific car
*   **lateral control**: Steering control
*   **longitudinal control**: Gas/Brakes control
*   **fingerprinting**: Automatic car identification
*   **LKAS**: Lane Keeping Assist System
*   **ACC**: Adaptive Cruise Control
*   **harness**: Car-specific hardware (e.g. from [comma.ai/shop/car-harness](https://comma.ai/shop/car-harness))
*   **panda**: Hardware for CAN bus access
*   **ECU**: Electronic Control Unit (car's computer)
*   **CAN bus**: Communication network for ECUs
*   **cabana**: Tool for reverse engineering CAN messages
*   **DBC file**: Contains CAN message definitions
*   **openpilot**: ADAS system supported by opendbc
*   **comma**: The company behind opendbc
*   **comma 3X**: Hardware to run openpilot

### More Resources

*   [*How Do We Control The Car?*](https://www.youtube.com/watch?v=nNU6ipme878&pp=ygUoY29tbWEgY29uIDIwMjEgaG93IGRvIHdlIGNvbnRyb2wgdGhlIGNhcg%3D%3D) by [@robbederks](https://github.com/robbederks) from COMMA_CON 2021
*   [*How to Port a Car*](https://www.youtube.com/watch?v=XxPS5TpTUnI&t=142s&pp=ygUPamFzb24gY29tbWEgY29u) by [@jyoung8607](https://github.com/jyoung8607) from COMMA_CON 2023
*   [commaCarSegments](https://huggingface.co/datasets/commaai/commaCarSegments): a massive dataset of CAN data from 300 different car models
*   [cabana](https://github.com/commaai/openpilot/tree/master/tools/cabana#readme): our tool for reverse engineering CAN messages
*   [can_print_changes.py](https://github.com/commaai/openpilot/blob/master/selfdrive/debug/can_print_changes.py): diff the whole CAN bus across two drives, such as one without any LKAS and one with LKAS
*   [longitudinal maneuvers](https://github.com/commaai/openpilot/tree/master/tools/longitudinal_maneuvers): a tool for evaluating and tuning longitudinal control
*   [opendbc data](https://commaai.github.io/opendbc-data/): a repository of longitudinal maneuver evaluations

## Join the Team!

comma is actively hiring engineers to contribute to opendbc and [openpilot](https://github.com/commaai/openpilot); apply at [comma.ai/jobs](https://comma.ai/jobs).