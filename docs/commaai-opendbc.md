<div align="center" style="text-align: center;">

<h1>opendbc: Your Gateway to Vehicle Control</h1>
<p>
  <b>Unlock the full potential of your car with opendbc, a Python API for in-depth vehicle control and data access.</b>
  <br>
  Control steering, gas, brakes, and more, all while reading crucial vehicle data like speed and steering angle.
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

## Introduction

opendbc empowers you to interact with your car's systems. Built for the community, it provides a robust Python API to control steering, gas, brakes, and access valuable vehicle data.  While integral to the [openpilot](https://github.com/commaai/openpilot) project, opendbc expands its reach by reading and writing to EV charge status, and more, allowing for a comprehensive vehicle management experience.

## Key Features

*   **Control**: Take command of your car's steering, acceleration, and braking systems.
*   **Data Access**: Retrieve real-time data including speed, steering angle, and more.
*   **Broad Compatibility**: Designed to support the majority of vehicles equipped with LKAS and ACC systems, starting from 2016.
*   **Community-Driven**:  Benefit from the open-source community's contributions, documentation, and support.
*   **Extensible**: Adapt and extend opendbc to meet your project needs, including EV charge status and door control.

## Quick Start

Get started with opendbc by cloning the repository and running the test script:

```bash
git clone https://github.com/commaai/opendbc.git
cd opendbc
./test.sh
```

This script handles dependency installation, building, linting, and testing.  For more granular control, you can run the individual commands:

```bash
pip3 install -e .[testing,docs]  # install dependencies
scons -j8                        # build with 8 cores
pytest .                         # run the tests
lefthook run lint                # run the linter
```

Explore the `examples/` directory for ready-to-use programs, including `examples/joystick.py` to control your car with a joystick.

## Project Structure

*   `opendbc/dbc/`:  Stores [DBC](https://en.wikipedia.org/wiki/CAN_bus#DBC_(CAN_Database_Files)) files for your car's data.
*   `opendbc/can/`:  A library to parse and build CAN messages using DBC files.
*   `opendbc/car/`:  High-level Python library to interact with cars.
*   `opendbc/safety/`:  Ensures functional safety for supported cars.

## Contributing

Join the opendbc community! Development is coordinated on [GitHub](https://github.com/commaai/opendbc). Get involved by visiting the `#dev-opendbc-cars` channel on [Discord](https://discord.comma.ai).

## How to Port a Car

The steps to add support for a new car or enhance existing models are detailed within the documentation, and are well-documented [here](https://github.com/commaai/opendbc/blob/docs/README.md#how-to-port-a-car).

## Roadmap

*   [ ]  `pip install opendbc` support
*   [ ]  100% type and line coverage
*   [ ]  Simplify car port creation with tools, tests, and documentation.
*   [ ]  Enhance car support for all vehicles with LKAS and ACC interfaces.
*   [ ]  Develop automatic lateral and longitudinal control/tuning evaluation.
*   [ ]  Implement auto-tuning features for lateral and longitudinal control.
*   [ ]  Integrate Automatic Emergency Braking support.

## Safety Model

opendbc's safety firmware, when used with a [panda](https://comma.ai/shop/panda), defaults to `SAFETY_SILENT` mode for secure operation. Safety modes provide message control based on customizable states.

The safety firmware prioritizes rigorous code quality, and enforces these standards:

*   Static code analysis via [cppcheck](https://github.com/danmar/cppcheck/).
*   [MISRA C:2012](https://misra.org.uk/) compliance checks.
*   Strict compiler flags.
*   Comprehensive [unit tests](opendbc/safety/tests) with 100% line coverage.
*   Mutation testing for the safety unit tests.
*   Ruff linter and [mypy](https://mypy-lang.org/) on the car interface library.

## Bounties

The community is rewarded for their contributions.  Bounties include:

*   $2000 - [Any car brand / platform port](https://github.com/orgs/commaai/projects/26/views/1?pane=issue&itemId=47913774)
*   $250 - [Any car model port](https://github.com/orgs/commaai/projects/26/views/1?pane=issue&itemId=47913790)
*   $300 - [Reverse Engineering a new Actuation Message](https://github.com/orgs/commaai/projects/26/views/1?pane=issue&itemId=73445563)

See [comma.ai/bounties](comma.ai/bounties) for more opportunities.

## FAQ

*   **How do I use this?**  Use it with a [comma 3X](https://comma.ai/shop/comma-3x).
*   **Which cars are supported?**  Check the [supported cars list](docs/CARS.md).
*   **Can I add support for my car?** Yes! Refer to the guide [here](https://github.com/commaai/opendbc/blob/docs/README.md#how-to-port-a-car).
*   **Which cars can be supported?** Any car with LKAS and ACC support. Find more info [here](https://github.com/commaai/openpilot/blob/master/docs/CARS.md#dont-see-your-car-here).
*   **How does this work?** This allows you to replace your car's built-in features.  See [this talk](https://www.youtube.com/watch?v=FL8CxUSfipM) for more information.
*   **Is there a timeline or roadmap for adding car support?** Community driven, with comma doing final safety and quality validation.

## Terms

A comprehensive list of terms is found [here](https://github.com/commaai/opendbc/blob/docs/README.md#terms) and below:

*   **port**: integrating a car.
*   **lateral control**: steering.
*   **longitudinal control**: acceleration/braking.
*   **fingerprinting**: identifying cars automatically.
*   **[LKAS](https://en.wikipedia.org/wiki/Lane_departure_warning_system)**: lane keeping assist.
*   **[ACC](https://en.wikipedia.org/wiki/Adaptive_cruise_control)**: adaptive cruise control.
*   **[harness](https://comma.ai/shop/car-harness)**: car-specific ADAS hardware.
*   **[panda](https://github.com/commaai/panda)**: hardware to access CAN bus.
*   **[ECU](https://en.wikipedia.org/wiki/Electronic_control_unit)**: car computer modules.
*   **[CAN bus](https://en.wikipedia.org/wiki/CAN_bus)**: a car's communication bus.
*   **[cabana](https://github.com/commaai/openpilot/tree/master/tools/cabana#readme)**: tool to reverse engineer CAN messages.
*   **[DBC file](https://en.wikipedia.org/wiki/CAN_bus#DBC)**: defines CAN message data.
*   **[openpilot](https://github.com/commaai/openpilot)**: an ADAS system.
*   **[comma](https://github.com/commaai)**: the company behind opendbc.
*   **[comma 3X](https://comma.ai/shop/comma-3x)**: the hardware used to run openpilot.

## More Resources

*   [*How Do We Control The Car?*](https://www.youtube.com/watch?v=nNU6ipme878&pp=ygUoY29tbWEgY29uIDIwMjEgaG93IGRvIHdlIGNvbnRyb2wgdGhlIGNhcg%3D%3D) by [@robbederks](https://github.com/robbederks) from COMMA_CON 2021
*   [*How to Port a Car*](https://www.youtube.com/watch?v=XxPS5TpTUnI&t=142s&pp=ygUPamFzb24gY29tbWEgY29u) by [@jyoung8607](https://github.com/jyoung8607) from COMMA_CON 2023
*   [commaCarSegments](https://huggingface.co/datasets/commaai/commaCarSegments): a massive dataset of CAN data from 300 different car models
*   [cabana](https://github.com/commaai/openpilot/tree/master/tools/cabana#readme): tool for reverse engineering CAN messages
*   [can_print_changes.py](https://github.com/commaai/openpilot/blob/master/selfdrive/debug/can_print_changes.py): diff the whole CAN bus across two drives
*   [longitudinal maneuvers](https://github.com/commaai/openpilot/tree/master/tools/longitudinal_maneuvers): a tool for evaluating and tuning longitudinal control
*   [opendbc data](https://commaai.github.io/opendbc-data/): a repository of longitudinal maneuver evaluations

## Join the Team!

[comma.ai](https://comma.ai/jobs) is hiring talented engineers.