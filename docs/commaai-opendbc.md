<div align="center">
  <h1>opendbc</h1>
  <p><b>opendbc: Open-Source Car Control - Unlock Your Car's Potential.</b></p>
  <p>
    Gain control over your car's steering, gas, and brakes with this powerful Python API.  Read speed, steering angle, and more.
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

opendbc is a Python API allowing you to control the steering, gas, and brakes of your car, providing access to a wide range of vehicle data.  Built to support the openpilot project, it offers extensive functionality for interacting with modern vehicles.  [See the original repository on GitHub](https://github.com/commaai/opendbc).

**Key Features:**

*   **Full Car Control:** Manipulate steering, gas, and brakes.
*   **Real-time Data Access:** Read crucial vehicle data such as speed and steering angle.
*   **Extensive Vehicle Support:** Aims to support numerous vehicles with LKAS and ACC systems.
*   **Open Source & Community Driven:** Benefit from collaborative development and contributions.
*   **Comprehensive Documentation:** Detailed guides and resources for usage and contribution.

## Quick Start

```bash
git clone https://github.com/commaai/opendbc.git
cd opendbc

# Comprehensive installation, build, and testing with one command. This runs in CI.
./test.sh

# Individual commands for more control
pip3 install -e .[testing,docs]  # Install dependencies
scons -j8                        # Build with 8 cores
pytest .                         # Run tests
lefthook run lint                # Run linter
```

[`examples/`](examples/) contains example programs to help you get started reading and controlling your car.
[`examples/joystick.py`](examples/joystick.py) allows you to control a car with a joystick.

## Project Structure

*   [`opendbc/dbc/`](opendbc/dbc/) - Contains [DBC](https://en.wikipedia.org/wiki/CAN_bus#DBC) files for different vehicles.
*   [`opendbc/can/`](opendbc/can/) - Library for parsing and building CAN messages from DBC files.
*   [`opendbc/car/`](opendbc/car/) - High-level Python library for interacting with cars.
*   [`opendbc/safety/`](opendbc/safety/) - Functional safety mechanisms for supported vehicles.

## How to Port a Car

This detailed guide walks you through the process of adding support for new cars or improving existing ones.

### Connect to the Car

Requires a comma 3X and compatible car harness (available at comma.ai/shop).

### Structure of a Port

Car ports are located in `opendbc/car/<brand>/` and contain:

*   `carstate.py`: Parses CAN data.
*   `carcontroller.py`: Sends CAN messages.
*   `<brand>can.py`: Helper functions for CAN message building.
*   `fingerprints.py`: ECU firmware database.
*   `interface.py`: High-level interface class.
*   `radar_interface.py`: Radar parsing.
*   `values.py`: Supported car enumeration.

### Reverse Engineer CAN messages

Use [cabana](https://github.com/commaai/openpilot/tree/master/tools/cabana) to analyze recorded CAN data.

### Tuning

Use the [longitudinal maneuvers](https://github.com/commaai/openpilot/tree/master/tools/longitudinal_maneuvers) report to assess and tune your car's longitudinal control.

## Contributing

Contributions are welcome!  Coordinate via GitHub and [Discord](https://discord.comma.ai) (see `#dev-opendbc-cars`).

### Roadmap

**Short Term:**

*   `pip install opendbc`
*   100% type and line coverage.
*   Improve car port development process (refactors, tools, etc.).
*   Improve state reporting for supported cars. (https://github.com/commaai/opendbc/issues/1144)

**Longer Term:**

*   Expand support to all LKAS + ACC equipped vehicles.
*   Automated lateral and longitudinal control/tuning evaluation.
*   Auto-tuning for lateral and longitudinal control.
*   Automatic Emergency Braking (AEB) support.

## Safety Model

opendbc's safety firmware runs with a [panda](https://comma.ai/shop/panda) and is by default in `SAFETY_SILENT` mode.  Safety modes support `controls_allowed` for message filtering based on board state.

## Code Rigor

High standards are maintained for the application code rigor within the `safety` folder.  Includes:

*   Static code analysis with [cppcheck](https://github.com/danmar/cppcheck/) (including MISRA C:2012 checks).
*   Strict compiler flags (-Wall, -Wextra, etc.).
*   Unit tests for safety logic, including mutation tests.
*   Ruff linter and mypy for the car interface library.

### Bounties

Earn bounties for contributions!

*   $2000 - [Any car brand / platform port](https://github.com/orgs/commaai/projects/26/views/1?pane=issue&itemId=47913774)
*   $250 - [Any car model port](https://github.com/orgs/commaai/projects/26/views/1?pane=issue&itemId=47913790)
*   $300 - [Reverse Engineering a new Actuation Message](https://github.com/orgs/commaai/projects/26/views/1?pane=issue&itemId=73445563)

Also, higher value bounties for popular cars are available at [comma.ai/bounties](comma.ai/bounties).

## FAQ

*   **How do I use this?** Use with a [comma 3X](https://comma.ai/shop/comma-3x).
*   **Which cars are supported?** See the [supported cars list](docs/CARS.md).
*   **Can I add support for my car?** Yes! Read the guide [here](https://github.com/commaai/opendbc/blob/docs/README.md#how-to-port-a-car).
*   **Which cars can be supported?** Cars with LKAS and ACC. More info [here](https://github.com/commaai/openpilot/blob/master/docs/CARS.md#dont-see-your-car-here).
*   **How does this work?** Hardware replaces the lane keep and adaptive cruise features.  See [this talk](https://www.youtube.com/watch?v=FL8CxUSfipM) for a detailed explanation.
*   **Is there a timeline or roadmap for adding car support?** Community-driven; comma performs final validation.

### Terms

*   **port**: Car integration and support.
*   **lateral control**: Steering control.
*   **longitudinal control**: Gas/brake control.
*   **fingerprinting**: Automatic car identification.
*   **LKAS**: Lane Keeping Assist.
*   **ACC**: Adaptive Cruise Control.
*   **harness**: Car-specific hardware to connect to the ADAS messages.
*   **panda**: Hardware for CAN bus access.
*   **ECU**: Electronic Control Unit (car computer).
*   **CAN bus**: Car's internal communication network.
*   **cabana**: Tool for reverse engineering CAN messages.
*   **DBC file**: Message definitions for a CAN bus.
*   **openpilot**: ADAS system supported by opendbc.
*   **comma**: The company behind opendbc.
*   **comma 3X**: Hardware to run openpilot.

### More resources

*   [*How Do We Control The Car?*](https://www.youtube.com/watch?v=nNU6ipme878&pp=ygUoY29tbWEgY29uIDIwMjEgaG93IGRvIHdlIGNvbnRyb2wgdGhlIGNhcg%3D%3D) by [@robbederks](https://github.com/robbederks) from COMMA_CON 2021
*   [*How to Port a Car*](https://www.youtube.com/watch?v=XxPS5TpTUnI&t=142s&pp=ygUPamFzb24gY29tbWEgY29u) by [@jyoung8607](https://github.com/jyoung8607) from COMMA_CON 2023
*   [commaCarSegments](https://huggingface.co/datasets/commaai/commaCarSegments): a massive dataset of CAN data from 300 different car models
*   [cabana](https://github.com/commaai/openpilot/tree/master/tools/cabana#readme): our tool for reverse engineering CAN messages
*   [can_print_changes.py](https://github.com/commaai/openpilot/blob/master/selfdrive/debug/can_print_changes.py): diff the whole CAN bus across two drives, such as one without any LKAS and one with LKAS
*   [longitudinal maneuvers](https://github.com/commaai/openpilot/tree/master/tools/longitudinal_maneuvers): a tool for evaluating and tuning longitudinal control
*   [opendbc data](https://commaai.github.io/opendbc-data/): a repository of longitudinal maneuver evaluations

## Come work with us -- [comma.ai/jobs](https://comma.ai/jobs)

Join the comma team and contribute to opendbc and [openpilot](https://github.com/commaai/openpilot). We welcome contributions!