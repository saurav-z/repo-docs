# opendbc: Your Python API for Vehicle Control and Data Access

**Take control of your car's steering, gas, brakes, and more with opendbc, a powerful Python API.**  [View the original repository on GitHub](https://github.com/commaai/opendbc).

<div align="center">
  <a href="https://docs.comma.ai">Docs</a>
  <span> · </span>
  <a href="https://github.com/commaai/openpilot/blob/master/docs/CONTRIBUTING.md">Contribute</a>
  <span> · </span>
  <a href="https://discord.comma.ai">Discord</a>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![X Follow](https://img.shields.io/twitter/follow/comma_ai)](https://x.com/comma_ai)
[![Discord](https://img.shields.io/discord/469524606043160576)](https://discord.comma.ai)
</div>

---

opendbc empowers you to interact with your car's electronic systems, enabling advanced driver-assistance system (ADAS) features and building custom vehicle management applications. It leverages the existing capabilities of modern vehicles with features such as Lane Keeping Assist (LKAS) and Adaptive Cruise Control (ACC).

**Key Features:**

*   **Control:** Access and control steering, gas, and brakes.
*   **Data Acquisition:** Read speed, steering angle, and other critical vehicle data.
*   **Extensive Support:** Designed to support a wide range of vehicles with LKAS and ACC.
*   **Open Source:** Benefit from a community-driven project with active development and contributions.
*   **ADAS Integration:**  Primary focus on supporting ADAS interfaces for openpilot.

## Getting Started

1.  **Clone the Repository:** `git clone https://github.com/commaai/opendbc.git`
2.  **Navigate to the Directory:** `cd opendbc`
3.  **Run the Test Script:** `./test.sh` (This installs dependencies, builds, runs tests, and lints your code.)

    *   This script runs the following commands:
        *   `pip3 install -e .[testing,docs]`  (Install dependencies)
        *   `scons -j8` (Build with 8 cores)
        *   `pytest .` (Run tests)
        *   `lefthook run lint` (Run the linter)

4.  **Explore Examples:** The `examples/` directory contains example programs, including `examples/joystick.py` to control your car with a joystick.

## Project Structure

*   `opendbc/dbc/`: Contains [DBC](https://en.wikipedia.org/wiki/CAN_bus#DBC) files defining CAN message specifications.
*   `opendbc/can/`: Provides a library for parsing and building CAN messages from DBC files.
*   `opendbc/car/`: A high-level Python library for interacting with cars.
*   `opendbc/safety/`: Contains functional safety mechanisms.

## Contributing and Car Porting

[Contribute to opendbc](https://github.com/commaai/openpilot/blob/master/docs/CONTRIBUTING.md) and help expand the supported car list. This project thrives on community contributions.

### How to Port a Car

Follow the comprehensive guide within the documentation to add support for your vehicle.  The process typically involves connecting to the car, reverse engineering CAN messages using tools like Cabana, and tuning the control parameters.

### Roadmap & Bounties

Explore the roadmap for upcoming features and the bounty program for contributing.  Reward yourself for contributing to opendbc!

## Safety Model

The opendbc safety firmware is essential when paired with the [panda](https://comma.ai/shop/panda), as used for openpilot, providing safety.  This safety logic is tested through regression tests and is held to high standards, with thorough code analysis, including:
*   Static code analysis with [cppcheck](https://github.com/danmar/cppcheck/)
*   [MISRA C:2012](https://misra.org.uk/) compliance checks using [cppcheck](https://github.com/danmar/cppcheck/)
*   Compiler options with strict settings, including `-Wall -Wextra -Wstrict-prototypes -Werror`
*   Unit tests and mutation testing of the safety unit tests
*   Linting with [ruff linter](https://github.com/astral-sh/ruff) and [mypy](https://mypy-lang.org/)

### Bounties
*   $2000 - [Any car brand / platform port](https://github.com/orgs/commaai/projects/26/views/1?pane=issue&itemId=47913774)
*   $250 - [Any car model port](https://github.com/orgs/commaai/projects/26/views/1?pane=issue&itemId=47913790)
*   $300 - [Reverse Engineering a new Actuation Message](https://github.com/orgs/commaai/projects/26/views/1?pane=issue&itemId=73445563)

In addition to the standard bounties, we also offer higher value bounties for more popular cars. See those at [comma.ai/bounties](comma.ai/bounties).

## FAQ

**How do I use this?** A [comma 3X](https://comma.ai/shop/comma-3x) is custom-designed to be the best way to run and develop opendbc and openpilot.

**Which cars are supported?** See the [supported cars list](docs/CARS.md).

**Can I add support for my car?** Yes, most car support comes from the community. Read the guide [here](https://github.com/commaai/opendbc/blob/docs/README.md#how-to-port-a-car).

**Which cars can be supported?** Any car with LKAS and ACC. More info [here](https://github.com/commaai/openpilot/blob/master/docs/CARS.md#dont-see-your-car-here).

**How does this work?** In short, we designed hardware to replace your car's built-in lane keep and adaptive cruise features. See [this talk](https://www.youtube.com/watch?v=FL8CxUSfipM) for an in-depth explanation.

**Is there a timeline or roadmap for adding car support?** No, most car support comes from the community, with comma doing final safety and quality validation. The more complete the community car port is and the more popular the car is, the more likely we are to pick it up as the next one to validate.

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

## Join the Team!

**[comma.ai/jobs](https://comma.ai/jobs):** comma is hiring engineers to work on opendbc and [openpilot](https://github.com/commaai/openpilot).  We value contributions and encourage you to apply!