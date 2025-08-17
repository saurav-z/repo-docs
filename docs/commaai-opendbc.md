<div align="center" style="text-align: center;">

<h1>opendbc: Open-Source Car Communication API</h1>
<p>
  <b>Unlock the power of your car's data with opendbc, a Python API for in-depth vehicle control.</b>
  <br>
  Control steering, acceleration, braking, and more. Read vital information like speed and steering angle.
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

opendbc is a powerful Python API that enables you to interact with your car's internal systems, providing control over critical functions and access to real-time data. This open-source project, primarily supporting ADAS interfaces for [openpilot](https://github.com/commaai/openpilot), aims to support a wide range of vehicles.

**Key Features:**

*   **Control:** Command steering, gas, and brakes.
*   **Data Acquisition:** Read speed, steering angle, and other crucial vehicle data.
*   **Open Source:** Leverage a community-driven project with extensive documentation and support.
*   **Vehicle Compatibility:** Designed to support a growing number of vehicles with LKAS and ACC.
*   **Extensible:** Contribute to expand functionality and vehicle support.

Ready to dive in? Check out the original repository at [https://github.com/commaai/opendbc](https://github.com/commaai/opendbc).

---

## Getting Started

This section provides a quick overview of how to set up and run opendbc.

### Installation

1.  Clone the repository:

    ```bash
    git clone https://github.com/commaai/opendbc.git
    cd opendbc
    ```

2.  Run the test script (installs dependencies, builds, lints, and tests):

    ```bash
    ./test.sh
    ```

    Alternatively, use the individual commands:

    ```bash
    pip3 install -e .[testing,docs]  # install dependencies
    scons -j8                        # build with 8 cores
    pytest .                         # run the tests
    lefthook run lint                # run the linter
    ```

### Examples

Explore example programs in the `examples/` directory.  For instance, [`examples/joystick.py`](examples/joystick.py) demonstrates how to control a car with a joystick.

## Project Structure

*   [`opendbc/dbc/`](opendbc/dbc/) - Contains [DBC](https://en.wikipedia.org/wiki/CAN_bus#DBC) files (database files defining CAN message structures).
*   [`opendbc/can/`](opendbc/can/) - Library for parsing and building CAN messages from DBC files.
*   [`opendbc/car/`](opendbc/car/) - High-level Python library for interacting with cars.
*   [`opendbc/safety/`](opendbc/safety/) - Functional safety components for supported cars.

## How to Port a Car

This guide provides detailed instructions for adding support for new vehicles or improving existing ones.

### Connecting to the Car

1.  Connect to the car using a [comma 3X](https://comma.ai/shop/comma-3x) and a car harness.
2.  The harness connects to two CAN buses and allows for actuation message injection.
3.  Pre-made harnesses are available at comma.ai/shop, or you can create your own using a developer harness.

### Port Structure

A typical car port, located in `opendbc/car/<brand>/`, includes:

*   `carstate.py`: Parses CAN data.
*   `carcontroller.py`: Sends control messages.
*   `<brand>can.py`: Helper functions for building CAN messages.
*   `fingerprints.py`: ECU firmware identification.
*   `interface.py`: High-level interface class.
*   `radar_interface.py`: Radar data parsing (if applicable).
*   `values.py`: Defines supported car models.

### Reverse Engineering CAN Messages

1.  Record a driving session with various events (LKAS, ACC, steering, etc.).
2.  Use [cabana](https://github.com/commaai/openpilot/tree/master/tools/cabana) to analyze the recorded data and identify relevant CAN messages.

### Tuning

#### Longitudinal Control

Use the [longitudinal maneuvers](https://github.com/commaai/openpilot/tree/master/tools/longitudinal_maneuvers) report to fine-tune longitudinal control performance.

## Contributing

Join the opendbc community on GitHub and [Discord](https://discord.comma.ai). Check the `#dev-opendbc-cars` channel for collaborative development and the `Vehicle Specific` section on Discord.

### Roadmap

*   \[ ] `pip install opendbc`
*   \[ ] 100% type coverage
*   \[ ] 100% line coverage
*   \[ ] Improve car port usability (refactors, tools, tests, and docs)
*   \[ ] Enhance car state information (https://github.com/commaai/opendbc/issues/1144)
*   \[ ] Expand support to all cars with LKAS + ACC.
*   \[ ] Automate lateral and longitudinal control evaluation.
*   \[ ] Implement auto-tuning for lateral and longitudinal control.
*   \[ ] Integrate Automatic Emergency Braking.

## Safety Model

opendbc utilizes a robust safety model when paired with a [panda](https://comma.ai/shop/panda). The default `SAFETY_SILENT` mode ensures bus silence until a safety mode is selected.  Safety modes can incorporate `controls_allowed` to enable or disable messages based on defined conditions.

## Code Rigor

The `safety` folder employs stringent application code standards, essential for its critical role in conjunction with [openpilot](https://github.com/commaai/openpilot) and [panda](https://github.com/commaai/panda).

*   Static code analysis is performed using [cppcheck](https://github.com/danmar/cppcheck/), including a [MISRA C:2012](https://misra.org.uk/) addon.
*   Compiler flags: `-Wall -Wextra -Wstrict-prototypes -Werror`.
*   [Unit tests](opendbc/safety/tests) are implemented for each supported car variant.
*   Mutation testing on MISRA coverage and 100% line coverage enforced on safety tests.
*   The [ruff linter](https://github.com/astral-sh/ruff) and [mypy](https://mypy-lang.org/) are utilized on the car interface library.

### Bounties

Earn rewards for contributing:

*   $2000 - [Any car brand / platform port](https://github.com/orgs/commaai/projects/26/views/1?pane=issue&itemId=47913774)
*   $250 - [Any car model port](https://github.com/orgs/commaai/projects/26/views/1?pane=issue&itemId=47913790)
*   $300 - [Reverse Engineering a new Actuation Message](https://github.com/orgs/commaai/projects/26/views/1?pane=issue&itemId=73445563)

Additional bounties for popular car models are available at [comma.ai/bounties](comma.ai/bounties).

## FAQ

*   **How do I use this?** A [comma 3X](https://comma.ai/shop/comma-3x) is designed for opendbc and openpilot development.
*   **Which cars are supported?** See the [supported cars list](docs/CARS.md).
*   **Can I add support for my car?** Yes, the community drives most car support. See the porting guide.
*   **Which cars can be supported?** Any car with LKAS and ACC. More info [here](https://github.com/commaai/openpilot/blob/master/docs/CARS.md#dont-see-your-car-here).
*   **How does this work?** Hardware is used to replace your car's lane keep and adaptive cruise features. See [this talk](https://www.youtube.com/watch?v=FL8CxUSfipM).
*   **Is there a timeline or roadmap for adding car support?** Community-driven, with comma conducting final safety validation.

### Glossary

*   **port**: Implementing support for a specific car.
*   **lateral control**: Steering control.
*   **longitudinal control**: Gas/brakes control.
*   **fingerprinting**: Automatic car identification.
*   **[LKAS](https://en.wikipedia.org/wiki/Lane_departure_warning_system)**: Lane Keeping Assist.
*   **[ACC](https://en.wikipedia.org/wiki/Adaptive_cruise_control)**: Adaptive Cruise Control.
*   **[harness](https://comma.ai/shop/car-harness)**: Hardware for connecting to the car's ADAS systems.
*   **[panda](https://github.com/commaai/panda)**: Hardware for CAN bus communication.
*   **[ECU](https://en.wikipedia.org/wiki/Electronic_control_unit)**: Car's electronic control units.
*   **[CAN bus](https://en.wikipedia.org/wiki/CAN_bus)**: Network connecting ECUs.
*   **[cabana](https://github.com/commaai/openpilot/tree/master/tools/cabana#readme)**: Tool for reverse engineering CAN messages.
*   **[DBC file](https://en.wikipedia.org/wiki/CAN_bus#DBC)**: Contains CAN message definitions.
*   **[openpilot](https://github.com/commaai/openpilot)**: ADAS system leveraging opendbc.
*   **[comma](https://github.com/commaai)**: The company behind opendbc.
*   **[comma 3X](https://comma.ai/shop/comma-3x)**: Hardware used to run openpilot.

### More Resources

*   [*How Do We Control The Car?*](https://www.youtube.com/watch?v=nNU6ipme878&pp=ygUoY29tbWEgY29uIDIwMjEgaG93IGRvIHdlIGNvbnRyb2wgdGhlIGNhcg%3D%3D) by [@robbederks](https://github.com/robbederks) from COMMA_CON 2021
*   [*How to Port a Car*](https://www.youtube.com/watch?v=XxPS5TpTUnI&t=142s&pp=ygUPamFzb24gY29tbWEgY29u) by [@jyoung8607](https://github.com/jyoung8607) from COMMA_CON 2023
*   [commaCarSegments](https://huggingface.co/datasets/commaai/commaCarSegments): a massive dataset of CAN data from 300 different car models
*   [cabana](https://github.com/commaai/openpilot/tree/master/tools/cabana#readme): our tool for reverse engineering CAN messages
*   [can_print_changes.py](https://github.com/commaai/openpilot/blob/master/selfdrive/debug/can_print_changes.py): diff the whole CAN bus across two drives, such as one without any LKAS and one with LKAS
*   [longitudinal maneuvers](https://github.com/commaai/openpilot/tree/master/tools/longitudinal_maneuvers): a tool for evaluating and tuning longitudinal control
*   [opendbc data](https://commaai.github.io/opendbc-data/): a repository of longitudinal maneuver evaluations

## Join the Team!

Explore opportunities to work on opendbc and [openpilot](https://github.com/commaai/openpilot) at [comma.ai/jobs](https://comma.ai/jobs). Contributions from community members are highly valued.