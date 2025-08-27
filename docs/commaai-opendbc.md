<div align="center" style="text-align: center;">

<h1>opendbc: Python API for Your Car</h1>
<p>
  <b>Take control of your vehicle's systems with opendbc, the open-source Python API.</b>  Effortlessly control gas, brakes, and steering while reading vital data like speed and steering angle.
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

## Key Features

*   **Complete Control:**  Manipulate steering, gas, and brakes on supported vehicles.
*   **Real-time Data Access:**  Read critical car data including speed, steering angle, and more.
*   **Open-Source & Community Driven:** Built upon open-source principles with strong community support.
*   **ADAS Focus:** Designed to support and enhance Advanced Driver-Assistance Systems (ADAS) like openpilot.
*   **Extensive Car Support:**  Expandable to include a wide range of vehicles with LKAS and ACC systems.

[View the original repository](https://github.com/commaai/opendbc)

---

opendbc is a Python-based API designed to interact with your car's systems, providing a powerful tool for developers, researchers, and automotive enthusiasts. It allows you to read and write data from your car's CAN bus, enabling control of key functions like steering, acceleration, and braking. The project is primarily focused on supporting ADAS interfaces for [openpilot](https://github.com/commaai/openpilot), but it also supports reading and writing many other car systems.

## Getting Started

Clone the repository and install dependencies to begin:

```bash
git clone https://github.com/commaai/opendbc.git
cd opendbc

# Runs installation, build, linting, and tests:
./test.sh

# Individual commands
pip3 install -e .[testing,docs]
scons -j8
pytest .
lefthook run lint
```

Explore the [`examples/`](examples/) directory for sample programs to read car data and control vehicle functions.  The [`examples/joystick.py`](examples/joystick.py) script is a good starting point for controlling your car with a joystick.

## Project Structure

*   `opendbc/dbc/`: Contains [DBC](https://en.wikipedia.org/wiki/CAN_bus#DBC) files defining CAN message formats.
*   `opendbc/can/`: Provides a library for parsing and creating CAN messages from DBC files.
*   `opendbc/car/`: The high-level Python library for interacting with cars.
*   `opendbc/safety/`:  Implements functional safety measures for supported vehicles.

## How to Port a Car

Detailed documentation is provided to help you in the process of [porting a car](https://github.com/commaai/opendbc/blob/docs/README.md#how-to-port-a-car).

## Contributing

Contribute to opendbc development via GitHub and [Discord](https://discord.comma.ai) where the community is very active in the `#dev-opendbc-cars` channel and `Vehicle Specific` section.

## Roadmap

*   [ ] `pip install opendbc`
*   [ ] 100% type coverage
*   [ ] 100% line coverage
*   [ ] Make car ports easier: refactors, tools, tests, and docs
*   [ ] Expose the state of all supported cars better: https://github.com/commaai/opendbc/issues/1144

## Safety Model

The opendbc safety firmware, used with [openpilot](https://github.com/commaai/openpilot) and [panda](https://comma.ai/shop/panda), offers robust safety features.  It operates in a `SAFETY_SILENT` mode by default, and safety modes optionally support `controls_allowed` which provides customizable message control.

## Code Rigor

The `opendbc/safety` folder is held to the highest standards due to its critical function.  This includes:

*   Static code analysis with [cppcheck](https://github.com/danmar/cppcheck/) and MISRA C:2012 checks.
*   Strict compiler flags: `-Wall -Wextra -Wstrict-prototypes -Werror`.
*   Extensive unit tests with 100% line coverage, and mutation testing.
*   Ruff linter and mypy checks on the car interface library.

## Bounties

Active bounties are available for contributions:

*   $2000 - [Any car brand / platform port](https://github.com/orgs/commaai/projects/26/views/1?pane=issue&itemId=47913774)
*   $250 - [Any car model port](https://github.com/orgs/commaai/projects/26/views/1?pane=issue&itemId=47913790)
*   $300 - [Reverse Engineering a new Actuation Message](https://github.com/orgs/commaai/projects/26/views/1?pane=issue&itemId=73445563)

Additional high-value bounties are also offered for popular car models, view them at [comma.ai/bounties](comma.ai/bounties).

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

## Join the Team - [comma.ai/jobs](https://comma.ai/jobs)

comma is actively hiring engineers to work on opendbc and [openpilot](https://github.com/commaai/openpilot). The team welcomes contributions, so consider becoming part of the project!