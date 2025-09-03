<div align="center" style="text-align: center;">

<h1>opendbc: Your Gateway to Vehicle Control and Data</h1>
<p>
  <b>opendbc is a powerful Python API that empowers you to control and access data from your car's systems.</b>
  <br>
  Take command of steering, gas, brakes, and more.  Unlock real-time insights into speed, steering angle, and other vital information.
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

opendbc offers a comprehensive Python API designed to interface with your car's internal systems, enabling control of critical functions and access to valuable data. Designed for advanced drivers and developers, opendbc aims to support the control of steering, gas, and brakes on most cars since 2016, as they now include advanced driver-assistance systems like LKAS and ACC.

**<ins>Key Features:</ins>**

*   **Control:** Manage steering, acceleration, and braking.
*   **Data Access:** Read vehicle speed, steering angle, and other crucial information.
*   **Extensive Car Support:**  Effortlessly interact with a wide range of vehicles.
*   **Open Source and Community Driven:** Benefit from a collaborative environment and actively contribute to the project.
*   **Built-in Safety:** Includes `opendbc/safety` for safety and critical function monitoring.

[**Learn More and Contribute on GitHub**](https://github.com/commaai/opendbc)

---

This README and the [supported cars list](docs/CARS.md) are your primary resources for utilizing, contributing to, and extending the capabilities of opendbc.

## Quick Start

```bash
git clone https://github.com/commaai/opendbc.git
cd opendbc

# Install dependencies, build, run linting, and run tests.
./test.sh

# Individual commands for reference:
pip3 install -e .[testing,docs]  # Install dependencies
scons -j8                        # Build with 8 cores
pytest .                         # Run the tests
lefthook run lint                # Run the linter
```

Example programs are located in the [`examples/`](examples/) directory, which allows you to read car data and control steering, gas, and brakes. For example, [`examples/joystick.py`](examples/joystick.py) allows you to control a car with a joystick.

### Project Structure

*   [`opendbc/dbc/`](opendbc/dbc/): Contains [DBC](https://en.wikipedia.org/wiki/CAN_bus#DBC) files.
*   [`opendbc/can/`](opendbc/can/): Library for parsing and building CAN messages.
*   [`opendbc/car/`](opendbc/car/): High-level library for interacting with cars using Python.
*   [`opendbc/safety/`](opendbc/safety/): Functional safety features for supported cars.

## How to Port a Car

This guide will help you add support for new cars, or improve existing ones.

### Connect to the Car

You'll need a comma 3X and a car harness to connect to the car. If available, use a pre-made harness; otherwise, a developer harness and connector crimping will do the trick.

### Structure of a port

A car port is located in `opendbc/car/<brand>/`:

*   `carstate.py`: Parses information from the CAN stream using DBC files.
*   `carcontroller.py`: Outputs CAN messages to control the car.
*   `<brand>can.py`: Python helpers built using the DBC file to build CAN messages.
*   `fingerprints.py`: Firmware versions for car model identification.
*   `interface.py`: High-level class for interfacing with the car.
*   `radar_interface.py`: Parses out the radar.
*   `values.py`: Enumerates the supported cars.

### Reverse Engineer CAN messages

Start by recording a route using interesting events, such as LKAS and ACC. Then load the route up in [cabana](https://github.com/commaai/openpilot/tree/master/tools/cabana).

### Tuning

#### Longitudinal

Use the [longitudinal maneuvers](https://github.com/commaai/openpilot/tree/master/tools/longitudinal_maneuvers) report to evaluate your car's longitudinal control and tune it.

## Contributing

Development happens on GitHub and [Discord](https://discord.comma.ai).  Check the `#dev-opendbc-cars` channel.

### Roadmap

*   `pip install opendbc`
*   100% type coverage
*   100% line coverage
*   Make car ports easier: refactors, tools, tests, and docs
*   Expose the state of all supported cars better: https://github.com/commaai/opendbc/issues/1144

Longer term
*   Extend support to every car with LKAS + ACC interfaces
*   Automatic lateral and longitudinal control/tuning evaluation
*   Auto-tuning for [lateral](https://blog.comma.ai/090release/#torqued-an-auto-tuner-for-lateral-control) and longitudinal control
*   [Automatic Emergency Braking](https://en.wikipedia.org/wiki/Automated_emergency_braking_system)

## Safety Model

When a [panda](https://comma.ai/shop/panda) powers up with [opendbc safety firmware](opendbc/safety), by default it's in `SAFETY_SILENT` mode. While in `SAFETY_SILENT` mode, the CAN buses are forced to be silent. In order to send messages, you have to select a safety mode. Some of safety modes (for example `SAFETY_ALLOUTPUT`) are disabled in release firmwares. In order to use them, compile and flash your own build.

Safety modes optionally support `controls_allowed`, which allows or blocks a subset of messages based on a customizable state in the board.

## Code Rigor

The opendbc safety firmware is written for its use in conjunction with [openpilot](https://github.com/commaai/openpilot) and [panda](https://comma.ai/shop/panda). The safety firmware, through its safety model, provides and enforces the
[openpilot safety](https://github.com/commaai/openpilot/blob/master/docs/SAFETY.md). Due to its critical function, it's important that the application code rigor within the `safety` folder is held to high standards.

These are the [CI regression tests](https://github.com/commaai/opendbc/actions) we have in place:
* A generic static code analysis is performed by [cppcheck](https://github.com/danmar/cppcheck/).
* In addition, [cppcheck](https://github.com/danmar/cppcheck/) has a specific addon to check for [MISRA C:2012](https://misra.org.uk/) violations. See [current coverage](opendbc/safety/tests/misra/coverage_table).
* Compiler options are relatively strict: the flags `-Wall -Wextra -Wstrict-prototypes -Werror` are enforced.
* The [safety logic](opendbc/safety) is tested and verified by [unit tests](opendbc/safety/tests) for each supported car variant.

The above tests are themselves tested by:
* a [mutation test](opendbc/safety/tests/misra/test_mutation.py) on the MISRA coverage
* 100% line coverage enforced on the safety unit tests

In addition, we run the [ruff linter](https://github.com/astral-sh/ruff) and [mypy](https://mypy-lang.org/) on the car interface library.

### Bounties

Every car port is eligible for a bounty:
* $2000 - [Any car brand / platform port](https://github.com/orgs/commaai/projects/26/views/1?pane=issue&itemId=47913774)
* $250 - [Any car model port](https://github.com/orgs/commaai/projects/26/views/1?pane=issue&itemId=47913790)
* $300 - [Reverse Engineering a new Actuation Message](https://github.com/orgs/commaai/projects/26/views/1?pane=issue&itemId=73445563)

In addition to the standard bounties, we also offer higher value bounties for more popular cars. See those at [comma.ai/bounties](comma.ai/bounties).

## FAQ

***How do I use this?*** A [comma 3X](https://comma.ai/shop/comma-3x) is custom-designed to be the best way to run and develop opendbc and openpilot.

***Which cars are supported?*** See the [supported cars list](docs/CARS.md).

***Can I add support for my car?*** Yes, most car support comes from the community. Read the guide [here](https://github.com/commaai/opendbc/blob/docs/README.md#how-to-port-a-car).

***Which cars can be supported?*** Any car with LKAS and ACC. More info [here](https://github.com/commaai/openpilot/blob/master/docs/CARS.md#dont-see-your-car-here).

***How does this work?*** In short, we designed hardware to replace your car's built-in lane keep and adaptive cruise features. See [this talk](https://www.youtube.com/watch?v=FL8CxUSfipM) for an in-depth explanation.

***Is there a timeline or roadmap for adding car support?*** No, most car support comes from the community, with comma doing final safety and quality validation. The more complete the community car port is and the more popular the car is, the more likely we are to pick it up as the next one to validate.

### Terms

*   **port**: refers to the integration and support of a specific car
*   **lateral control**: aka steering control
*   **longitudinal control**: aka gas/brakes control
*   **fingerprinting**: automatic process for identifying the car
*   **[LKAS](https://en.wikipedia.org/wiki/Lane_departure_warning_system)**: lane keeping assist
*   **[ACC](https://en.wikipedia.org/wiki/Adaptive_cruise_control)**: adaptive cruise control
*   **[harness](https://comma.ai/shop/car-harness)**: car-specific hardware to attach to the car and intercept the ADAS messages
*   **[panda](https://github.com/commaai/panda)**: hardware used to get on a car's CAN bus
*   **[ECU](https://en.wikipedia.org/wiki/Electronic_control_unit)**: computers or control modules inside the car
*   **[CAN bus](https://en.wikipedia.org/wiki/CAN_bus)**: a bus that connects the ECUs in a car
*   **[cabana](https://github.com/commaai/openpilot/tree/master/tools/cabana#readme)**: our tool for reverse engineering CAN messages
*   **[DBC file](https://en.wikipedia.org/wiki/CAN_bus#DBC)**: contains definitions for messages on a CAN bus
*   **[openpilot](https://github.com/commaai/openpilot)**: an ADAS system for cars supported by opendbc
*   **[comma](https://github.com/commaai)**: the company behind opendbc
*   **[comma 3X](https://comma.ai/shop/comma-3x)**: the hardware used to run openpilot

### More resources

*   [*How Do We Control The Car?*](https://www.youtube.com/watch?v=nNU6ipme878&pp=ygUoY29tbWEgY29uIDIwMjEgaG93IGRvIHdlIGNvbnRyb2wgdGhlIGNhcg%3D%3D) by [@robbederks](https://github.com/robbederks) from COMMA_CON 2021
*   [*How to Port a Car*](https://www.youtube.com/watch?v=XxPS5TpTUnI&t=142s&pp=ygUPamFzb24gY29tbWEgY29u) by [@jyoung8607](https://github.com/jyoung8607) from COMMA_CON 2023
*   [commaCarSegments](https://huggingface.co/datasets/commaai/commaCarSegments): a massive dataset of CAN data from 300 different car models
*   [cabana](https://github.com/commaai/openpilot/tree/master/tools/cabana#readme): our tool for reverse engineering CAN messages
*   [can_print_changes.py](https://github.com/commaai/openpilot/blob/master/selfdrive/debug/can_print_changes.py): diff the whole CAN bus across two drives, such as one without any LKAS and one with LKAS
*   [longitudinal maneuvers](https://github.com/commaai/openpilot/tree/master/tools/longitudinal_maneuvers): a tool for evaluating and tuning longitudinal control
*   [opendbc data](https://commaai.github.io/opendbc-data/): a repository of longitudinal maneuver evaluations

## Come work with us -- [comma.ai/jobs](https://comma.ai/jobs)

comma is hiring engineers to work on opendbc and [openpilot](https://github.com/commaai/openpilot). We love hiring contributors.