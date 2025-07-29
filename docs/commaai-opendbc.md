<!-- markdownlint-disable MD041 -->
<div align="center">
  <h1>opendbc: Open Car Data for Enhanced Vehicle Control</h1>
  <p><b>opendbc unlocks your car's potential by providing a Python API to read and control vehicle systems.</b></p>

  <p>
    <a href="https://docs.comma.ai">Docs</a>
    <span> · </span>
    <a href="https://github.com/commaai/openpilot/blob/master/docs/CONTRIBUTING.md">Contribute</a>
    <span> · </span>
    <a href="https://discord.comma.ai">Discord</a>
  </p>

  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
  [![X Follow](https://img.shields.io/twitter/follow/comma_ai)](https://x.com/comma_ai)
  [![Discord](https://img.shields.io/discord/469524606043160576)](https://discord.comma.ai)
</div>

---

## Key Features

*   **Python API:** Easily access and control your car's systems with Python.
*   **Control Capabilities:** Manage steering, gas, brakes, and more.
*   **Data Acquisition:** Read vehicle data such as speed and steering angle.
*   **Extensive Car Support:** Designed to support a wide range of vehicles with LKAS and ACC.
*   **Community Driven:** Benefit from a collaborative project with contributions from the community.
*   **Safety Focused:** Rigorous testing and safety models ensure responsible vehicle control.

## Introduction

opendbc is a powerful Python API designed to interact with your car's internal systems. It enables you to control key functions like steering, acceleration, and braking, while also allowing you to read real-time data such as speed and steering angle. Originally developed to support ADAS interfaces for [openpilot](https://github.com/commaai/openpilot), opendbc aims to be the best vehicle management app by reading and writing as many car features as possible.

## Quick Start

Get started with opendbc by cloning the repository and running the test script:

```bash
git clone https://github.com/commaai/opendbc.git
cd opendbc
./test.sh
```

The `test.sh` script installs dependencies, builds the project, runs tests, and performs linting.

## Project Structure

*   `opendbc/dbc/`: Contains [DBC](https://en.wikipedia.org/wiki/CAN_bus#DBC) files, which define the structure of CAN messages.
*   `opendbc/can/`: Provides a library for parsing and building CAN messages from DBC files.
*   `opendbc/car/`: Offers a high-level Python interface for interacting with cars.
*   `opendbc/safety/`: Implements functional safety measures for all supported cars.

## How to Contribute

Join the opendbc community on GitHub and [Discord](https://discord.comma.ai). Explore the `#dev-opendbc-cars` channel and the "Vehicle Specific" section for collaborative development.

### Roadmap

*   [ ] Implement `pip install opendbc`.
*   [ ] Achieve 100% type and line coverage.
*   [ ] Simplify car porting with refactors, tools, and documentation.
*   [ ] Enhance car state exposure (see Issue #1144).
*   [ ] Expand support to every car with LKAS and ACC interfaces.
*   [ ] Implement automatic lateral and longitudinal control/tuning evaluation.
*   [ ] Develop auto-tuning capabilities for lateral and longitudinal control.

## How to Port a Car

Adding support for new cars involves connecting to the car, reverse engineering CAN messages, and tuning control systems.

### Connect to the Car

*   Use a comma 3X and a car harness.
*   If a compatible harness isn't available, use a developer harness and crimp on the required connector.

### Structure of a Port

*   `opendbc/car/<brand>/`: Contains the car-specific implementation.
    *   `carstate.py`: Parses data from the CAN stream.
    *   `carcontroller.py`: Sends CAN messages to control the car.
    *   `<brand>can.py`: Provides helpers to build CAN messages.
    *   `fingerprints.py`: Identifies car models.
    *   `interface.py`: High-level interface class.
    *   `radar_interface.py`: Parses radar data.
    *   `values.py`: Lists supported cars.

### Reverse Engineer CAN Messages

*   Record a route with various events using a comma device (e.g. enabling LKAS and ACC).
*   Analyze the recording in [cabana](https://github.com/commaai/openpilot/tree/master/tools/cabana).

### Tuning

*   Use the [longitudinal maneuvers](https://github.com/commaai/openpilot/tree/master/tools/longitudinal_maneuvers) report to assess and tune longitudinal control.

## Safety Model

The opendbc safety firmware, used with [openpilot](https://github.com/commaai/openpilot) and [panda](https://comma.ai/shop/panda), provides and enforces safety features. It operates in `SAFETY_SILENT` mode by default, and you must select a safety mode to send messages. Some safety modes, such as `SAFETY_ALLOUTPUT`, are disabled in release firmwares and require custom builds.

## Code Rigor

The safety firmware undergoes rigorous testing, including:

*   Static code analysis with [cppcheck](https://github.com/danmar/cppcheck/), including MISRA C:2012 checks.
*   Compiler flags: `-Wall -Wextra -Wstrict-prototypes -Werror`.
*   Unit tests for each car variant in `opendbc/safety/tests`.
*   Mutation tests on MISRA coverage and 100% line coverage on safety unit tests.
*   Use of the [ruff linter](https://github.com/astral-sh/ruff) and [mypy](https://mypy-lang.org/) on the car interface library.

## Bounties

Contribute and earn bounties!

*   $2000 - [Any car brand / platform port](https://github.com/orgs/commaai/projects/26/views/1?pane=issue&itemId=47913774)
*   $250 - [Any car model port](https://github.com/orgs/commaai/projects/26/views/1?pane=issue&itemId=47913790)
*   $300 - [Reverse Engineering a new Actuation Message](https://github.com/orgs/commaai/projects/26/views/1?pane=issue&itemId=73445563)

Higher-value bounties are available for popular cars (see [comma.ai/bounties](https://comma.ai/bounties)).

## FAQ

**How do I use this?** A [comma 3X](https://comma.ai/shop/comma-3x) is designed to run and develop opendbc and openpilot.

**Which cars are supported?** See the [supported cars list](docs/CARS.md).

**Can I add support for my car?** Yes, most car support comes from the community. Read the guide [here](https://github.com/commaai/opendbc/blob/docs/README.md#how-to-port-a-car).

**Which cars can be supported?** Any car with LKAS and ACC. More info [here](https://github.com/commaai/openpilot/blob/master/docs/CARS.md#dont-see-your-car-here).

**How does this work?**  We designed hardware to replace your car's built-in lane keep and adaptive cruise features. See [this talk](https://www.youtube.com/watch?v=FL8CxUSfipM) for an in-depth explanation.

**Is there a timeline or roadmap for adding car support?** No, most car support comes from the community, with comma doing final safety and quality validation.

### Terms

*   **port**:  Integration and support of a specific car
*   **lateral control**: Steering control
*   **longitudinal control**: Gas/brakes control
*   **fingerprinting**: Automatic car identification
*   **[LKAS](https://en.wikipedia.org/wiki/Lane_departure_warning_system)**: Lane keeping assist
*   **[ACC](https://en.wikipedia.org/wiki/Adaptive_cruise_control)**: Adaptive cruise control
*   **[harness](https://comma.ai/shop/car-harness)**: Car-specific hardware to attach to intercept ADAS messages
*   **[panda](https://github.com/commaai/panda)**: Hardware to connect to a car's CAN bus
*   **[ECU](https://en.wikipedia.org/wiki/Electronic_control_unit)**: Computers or control modules inside the car
*   **[CAN bus](https://en.wikipedia.org/wiki/CAN_bus)**: Bus that connects ECUs in a car
*   **[cabana](https://github.com/commaai/openpilot/tree/master/tools/cabana#readme)**: Tool for reverse engineering CAN messages
*   **[DBC file](https://en.wikipedia.org/wiki/CAN_bus#DBC)**: Defines messages on a CAN bus
*   **[openpilot](https://github.com/commaai/openpilot)**: ADAS system for cars supported by opendbc
*   **[comma](https://github.com/commaai)**: The company behind opendbc
*   **[comma 3X](https://comma.ai/shop/comma-3x)**: Hardware to run openpilot

### More Resources

*   [*How Do We Control The Car?*](https://www.youtube.com/watch?v=nNU6ipme878&pp=ygUoY29tbWEgY29uIDIwMjEgaG93IGRvIHdlIGNvbnRyb2wgdGhlIGNhcg%3D%3D) by [@robbederks](https://github.com/robbederks) from COMMA_CON 2021
*   [*How to Port a Car*](https://www.youtube.com/watch?v=XxPS5TpTUnI&t=142s&pp=ygUPamFzb24gY29tbWEgY29u) by [@jyoung8607](https://github.com/jyoung8607) from COMMA_CON 2023
*   [commaCarSegments](https://huggingface.co/datasets/commaai/commaCarSegments): a massive dataset of CAN data from 300 different car models
*   [cabana](https://github.com/commaai/openpilot/tree/master/tools/cabana#readme): our tool for reverse engineering CAN messages
*   [can_print_changes.py](https://github.com/commaai/openpilot/blob/master/selfdrive/debug/can_print_changes.py): diff the whole CAN bus across two drives, such as one without any LKAS and one with LKAS
*   [longitudinal maneuvers](https://github.com/commaai/openpilot/tree/master/tools/longitudinal_maneuvers): a tool for evaluating and tuning longitudinal control
*   [opendbc data](https://commaai.github.io/opendbc-data/): a repository of longitudinal maneuver evaluations

## Join Our Team!

comma is hiring engineers; explore opportunities at [comma.ai/jobs](https://comma.ai/jobs).