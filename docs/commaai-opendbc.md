<div align="center" style="text-align: center;">

<h1>opendbc: Your Python API for Automotive Control</h1>
<p>
  <b>Unlock the power to control and interact with your car's systems using a powerful Python API.</b>
  <br>
  Gain control over steering, acceleration, braking, and more, while reading crucial vehicle data.
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

## About opendbc: Your Gateway to Automotive Control

opendbc is a powerful Python API designed to give you direct control and insight into your vehicle's systems. With opendbc, you can not only control steering, gas, and brakes, but also read vital information such as speed and steering angle.  Leverage your car's systems for advanced automation and data analysis.

**[Visit the GitHub Repository for more details](https://github.com/commaai/opendbc)**

### Key Features:

*   **Control:**  Command your car's steering, throttle, and brakes.
*   **Data Acquisition:** Access critical vehicle data, including speed, steering angle, and more.
*   **CAN Bus Interface:** Interact with your car's CAN bus, the backbone of its electronic communication.
*   **DBC Files:** Utilize DBC files to interpret CAN messages effectively.
*   **Car Porting:** Add support for new car models.

### Quick Start

Get started quickly with these simple steps:

```bash
git clone https://github.com/commaai/opendbc.git
cd opendbc

# Install, build, and test with a single command (recommended)
./test.sh

# Individual commands for more control:
pip3 install -e .[testing,docs]  # install dependencies
scons -j8                        # build with 8 cores
pytest .                         # run the tests
lefthook run lint                # run the linter
```

Explore example programs in the [`examples/`](examples/) directory, including [`examples/joystick.py`](examples/joystick.py) for joystick-based control.

### Project Structure

*   [`opendbc/dbc/`](opendbc/dbc/):  Repository of DBC (CAN Database) files.
*   [`opendbc/can/`](opendbc/can/):  Library for parsing and constructing CAN messages from DBC files.
*   [`opendbc/car/`](opendbc/car/): High-level Python library for interfacing with vehicles.
*   [`opendbc/safety/`](opendbc/safety/): Functional safety components for supported cars.

## How to Port a Car

This comprehensive guide walks you through everything from adding support for new vehicles to enhancing existing ones.  Learn how to integrate your car and contribute to the community.

### Connect to the Car

Begin by connecting to your car using a comma 3X and a compatible car harness.  If no harness exists, you can create a custom one with a developer harness.

### Structure of a Port

Each car port, generally contained in `opendbc/car/<brand>/`, features:

*   `carstate.py`: Parses data from the CAN stream.
*   `carcontroller.py`: Sends control messages to the car.
*   `<brand>can.py`: Helper functions for building CAN messages.
*   `fingerprints.py`: Identifies car models.
*   `interface.py`:  High-level interface class.
*   `radar_interface.py`: Parses radar data (if available).
*   `values.py`: Defines supported car models.

### Reverse Engineer CAN Messages

Record routes with diverse events, then analyze the CAN data using tools like [cabana](https://github.com/commaai/openpilot/tree/master/tools/cabana).

### Tuning

Optimize your car's performance using the [longitudinal maneuvers](https://github.com/commaai/openpilot/tree/master/tools/longitudinal_maneuvers) report.

## Contributing

Join the opendbc community and contribute to the project! All development is coordinated on GitHub and [Discord](https://discord.comma.ai).
Check out the `#dev-opendbc-cars` channel and `Vehicle Specific` section.

### Roadmap

*   **Short Term:**
    *   `pip install opendbc` support
    *   100% type and line coverage
    *   Simplify car porting
    *   Improved car state information
*   **Longer Term:**
    *   Support for every car with LKAS + ACC
    *   Automated tuning and evaluation
    *   Auto-tuning for lateral and longitudinal control
    *   Automatic Emergency Braking

## Safety Model

The opendbc safety firmware, used with [openpilot](https://github.com/commaai/openpilot) and [panda](https://github.com/commaai/panda), operates in different safety modes. The default `SAFETY_SILENT` mode ensures the CAN buses are inactive. Other modes require custom builds.

### Code Rigor

The safety firmware enforces strict coding standards:

*   Static code analysis with [cppcheck](https://github.com/danmar/cppcheck/) and MISRA C:2012 checks.
*   Strict compiler flags: `-Wall -Wextra -Wstrict-prototypes -Werror`.
*   Unit tests for safety logic with 100% line coverage.
*   Mutation tests on MISRA coverage.
*   Ruff linter and mypy on car interface library.

### Bounties

Earn bounties for contributing to opendbc!

*   $2000 - [Any car brand / platform port](https://github.com/orgs/commaai/projects/26/views/1?pane=issue&itemId=47913774)
*   $250 - [Any car model port](https://github.com/orgs/commaai/projects/26/views/1?pane=issue&itemId=47913790)
*   $300 - [Reverse Engineering a new Actuation Message](https://github.com/orgs/commaai/projects/26/views/1?pane=issue&itemId=73445563)

Higher value bounties are offered for popular car models at [comma.ai/bounties](comma.ai/bounties).

## FAQ

*   **How do I use this?** The [comma 3X](https://comma.ai/shop/comma-3x) is designed for use with opendbc and openpilot.
*   **Which cars are supported?** View the [supported cars list](docs/CARS.md).
*   **Can I add support for my car?** Absolutely! Follow the [car porting guide](https://github.com/commaai/opendbc/blob/docs/README.md#how-to-port-a-car).
*   **Which cars can be supported?** Cars with LKAS and ACC interfaces. See more info [here](https://github.com/commaai/openpilot/blob/master/docs/CARS.md#dont-see-your-car-here).
*   **How does this work?** Hardware is designed to replace lane keep and adaptive cruise features. See this [talk](https://www.youtube.com/watch?v=FL8CxUSfipM).
*   **Is there a timeline or roadmap for adding car support?** Community contributions are the primary driver of car support.

### Terms

A helpful glossary of terms used throughout the project.

*   **port**:  Integrating a specific car.
*   **lateral control**: Steering control.
*   **longitudinal control**: Gas/brakes control.
*   **fingerprinting**: Automated car identification.
*   **LKAS**: Lane Keeping Assist.
*   **ACC**: Adaptive Cruise Control.
*   **harness**: Car-specific hardware.
*   **panda**:  Hardware for CAN bus access.
*   **ECU**: Car's control modules.
*   **CAN bus**: Car's communication network.
*   **cabana**: Tool for CAN message analysis.
*   **DBC file**: Defines CAN messages.
*   **openpilot**: ADAS system using opendbc.
*   **comma**: The company behind opendbc.
*   **comma 3X**: The hardware to run openpilot.

### More resources

*   [*How Do We Control The Car?*](https://www.youtube.com/watch?v=nNU6ipme878&pp=ygUoY29tbWEgY29uIDIwMjEgaG93IGRvIHdlIGNvbnRyb2wgdGhlIGNhcg%3D%3D) by [@robbederks](https://github.com/robbederks) from COMMA_CON 2021
*   [*How to Port a Car*](https://www.youtube.com/watch?v=XxPS5TpTUnI&t=142s&pp=ygUPamFzb24gY29tbWEgY29u) by [@jyoung8607](https://github.com/jyoung8607) from COMMA_CON 2023
*   [commaCarSegments](https://huggingface.co/datasets/commaai/commaCarSegments): a massive dataset of CAN data from 300 different car models
*   [cabana](https://github.com/commaai/openpilot/tree/master/tools/cabana#readme): our tool for reverse engineering CAN messages
*   [can_print_changes.py](https://github.com/commaai/openpilot/blob/master/selfdrive/debug/can_print_changes.py): diff the whole CAN bus across two drives, such as one without any LKAS and one with LKAS
*   [longitudinal maneuvers](https://github.com/commaai/openpilot/tree/master/tools/longitudinal_maneuvers): a tool for evaluating and tuning longitudinal control
*   [opendbc data](https://commaai.github.io/opendbc-data/): a repository of longitudinal maneuver evaluations

## Come work with us -- [comma.ai/jobs](https://comma.ai/jobs)

Join the comma team and contribute to opendbc and [openpilot](https://github.com/commaai/openpilot)!