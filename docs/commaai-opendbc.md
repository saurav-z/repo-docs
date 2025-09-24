<div align="center">
  <h1>opendbc: Your Open-Source Car Interface</h1>
  <p><b>Take control of your car's systems with opendbc, a Python API for interacting with vehicle networks.</b></p>
  <p>
    Control gas, brakes, and steering, and read crucial data like speed and steering angle to build advanced automotive applications.
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

opendbc empowers you to interact with your car's CAN bus, enabling control over essential functions and access to vital data. Designed primarily to support ADAS interfaces for [openpilot](https://github.com/commaai/openpilot), opendbc also aims to support a wide range of vehicle management applications.

## Key Features

*   **Comprehensive Control:** Control steering, gas, and brakes.
*   **Real-Time Data:** Read speed, steering angle, and other critical vehicle data.
*   **Broad Support:** Works with many cars, with a focus on vehicles with LKAS and ACC.
*   **Open Source:**  Leverage the power of open source and community contributions.
*   **Python API:** Easy-to-use Python API for developers.

[Link to Original Repo: https://github.com/commaai/opendbc](https://github.com/commaai/opendbc)

## Quick Start

Get started quickly with these commands:

```bash
git clone https://github.com/commaai/opendbc.git
cd opendbc

./test.sh # Installs dependencies, builds, lints, and runs tests.
```

Or run the commands individually:

```bash
pip3 install -e .[testing,docs]  # install dependencies
scons -j8                        # build with 8 cores
pytest .                         # run the tests
lefthook run lint                # run the linter
```

Explore the `examples/` directory for sample programs, including `examples/joystick.py`, which allows you to control a car with a joystick.

### Project Structure

*   `opendbc/dbc/`: Contains DBC files (CAN database files).
*   `opendbc/can/`: Library for parsing and building CAN messages from DBC files.
*   `opendbc/car/`: High-level Python library for car interaction.
*   `opendbc/safety/`: Functional safety for supported cars.

## How to Port a Car

This section guides you through the process of adding support for new cars or improving existing ones.

### Connect to the Car

Connect to your car using a comma 3X and a car harness. Compatible harnesses are available at [comma.ai/shop](https://comma.ai/shop), or create your own with a developer harness.

### Structure of a Port

A car port resides in `opendbc/car/<brand>/`:

*   `carstate.py`: Parses CAN data.
*   `carcontroller.py`: Outputs CAN messages.
*   `<brand>can.py`: Python helpers for DBC files.
*   `fingerprints.py`: ECU firmware database.
*   `interface.py`: High-level car interface.
*   `radar_interface.py`: Radar parsing.
*   `values.py`: Enumerates supported cars.

### Reverse Engineer CAN Messages

Record a route with varied events, then load the data in [cabana](https://github.com/commaai/openpilot/tree/master/tools/cabana).

### Tuning

Use the [longitudinal maneuvers](https://github.com/commaai/openpilot/tree/master/tools/longitudinal_maneuvers) report for tuning.

## Contributing

Contribute via GitHub and [Discord](https://discord.comma.ai) in the `#dev-opendbc-cars` channel.

### Roadmap

**Short Term**

*   [ ] `pip install opendbc`
*   [ ] 100% type coverage
*   [ ] 100% line coverage
*   [ ] Improve car port creation.
*   [ ] Improve state representation of all supported cars.

**Longer Term**

*   [ ] Support every car with LKAS + ACC.
*   [ ] Automatic lateral and longitudinal control/tuning evaluation.
*   [ ] Auto-tuning for lateral and longitudinal control.
*   [ ] Automatic Emergency Braking support.

## Safety Model

The safety firmware is in `SAFETY_SILENT` mode by default. Select a safety mode to send messages, compiled from your build.

## Code Rigor

opendbc uses high-standard code rigor within the `safety` folder.

*   Static code analysis by [cppcheck](https://github.com/danmar/cppcheck/).
*   MISRA C:2012 violations check.
*   Compiler flags: `-Wall -Wextra -Wstrict-prototypes -Werror`.
*   Unit tests.
*   Mutation tests.
*   100% line coverage on safety unit tests.
*   Ruff linter and mypy are used.

### Bounties

Earn bounties for contributing:

*   $2000 - [Any car brand / platform port](https://github.com/orgs/commaai/projects/26/views/1?pane=issue&itemId=47913774)
*   $250 - [Any car model port](https://github.com/orgs/commaai/projects/26/views/1?pane=issue&itemId=47913790)
*   $300 - [Reverse Engineering a new Actuation Message](https://github.com/orgs/commaai/projects/26/views/1?pane=issue&itemId=73445563)

Plus, higher value bounties for popular cars at [comma.ai/bounties](https://comma.ai/bounties).

## FAQ

**How do I use this?** Requires a [comma 3X](https://comma.ai/shop/comma-3x).

**Which cars are supported?** See the [supported cars list](docs/CARS.md).

**Can I add support for my car?** Yes, read the guide [here](https://github.com/commaai/opendbc/blob/docs/README.md#how-to-port-a-car).

**Which cars can be supported?** Any car with LKAS and ACC. More info [here](https://github.com/commaai/openpilot/blob/master/docs/CARS.md#dont-see-your-car-here).

**How does this work?** Designed hardware to replace your car's lane keep and adaptive cruise features. Watch [this talk](https://www.youtube.com/watch?v=FL8CxUSfipM).

**Timeline for adding car support?** Community-driven, with comma performing final validation.

### Terms

*   **port**: Car integration
*   **lateral control**: Steering control
*   **longitudinal control**: Gas/brakes control
*   **fingerprinting**: Car identification
*   **[LKAS](https://en.wikipedia.org/wiki/Lane_departure_warning_system)**: Lane keeping assist
*   **[ACC](https://en.wikipedia.org/wiki/Adaptive_cruise_control)**: Adaptive cruise control
*   **[harness](https://comma.ai/shop/car-harness)**: Car-specific hardware
*   **[panda](https://github.com/commaai/panda)**: Hardware for CAN bus access
*   **[ECU](https://en.wikipedia.org/wiki/Electronic_control_unit)**: Car's control modules
*   **[CAN bus](https://en.wikipedia.org/wiki/CAN_bus)**: Car's communication network
*   **[cabana](https://github.com/commaai/openpilot/tree/master/tools/cabana#readme)**: Reverse engineering tool
*   **[DBC file](https://en.wikipedia.org/wiki/CAN_bus#DBC)**: CAN message definitions
*   **[openpilot](https://github.com/commaai/openpilot)**: ADAS system
*   **[comma](https://github.com/commaai)**: The company
*   **[comma 3X](https://comma.ai/shop/comma-3x)**: The hardware

### More Resources

*   [*How Do We Control The Car?*](https://www.youtube.com/watch?v=nNU6ipme878&pp=ygUoY29tbWEgY29uIDIwMjEgaG93IGRvIHdlIGNvbnRyb2wgdGhlIGNhcg%3D%3D) by [@robbederks](https://github.com/robbederks)
*   [*How to Port a Car*](https://www.youtube.com/watch?v=XxPS5TpTUnI&t=142s&pp=ygUPamFzb24gY29tbWEgY29u) by [@jyoung8607](https://github.com/jyoung8607)
*   [commaCarSegments](https://huggingface.co/datasets/commaai/commaCarSegments): CAN data from 300+ car models
*   [cabana](https://github.com/commaai/openpilot/tree/master/tools/cabana#readme): CAN reverse engineering
*   [can\_print\_changes.py](https://github.com/commaai/openpilot/blob/master/selfdrive/debug/can_print_changes.py): CAN bus diff
*   [longitudinal maneuvers](https://github.com/commaai/openpilot/tree/master/tools/longitudinal_maneuvers): Longitudinal control tuning
*   [opendbc data](https://commaai.github.io/opendbc-data/): Longitudinal maneuver evaluations

## Join the Team

Explore job opportunities at [comma.ai/jobs](https://comma.ai/jobs) and contribute to opendbc and [openpilot](https://github.com/commaai/openpilot).