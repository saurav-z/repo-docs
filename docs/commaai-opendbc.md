<div align="center" style="text-align: center;">

<h1>opendbc</h1>
<p>
  <b>Unlock your car's potential: opendbc is a powerful Python API for controlling and reading data from your vehicle.</b>
  <br>
  Gain control of steering, gas, brakes, and more, enabling advanced vehicle management and ADAS development.
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

**[Visit the original repository](https://github.com/commaai/opendbc) for more details.**

opendbc empowers developers and automotive enthusiasts to interact with modern vehicles' advanced driver-assistance systems (ADAS). This project enables precise control over crucial car functions.

**Key Features:**

*   **Control:** Manipulate steering, acceleration, and braking systems.
*   **Data Acquisition:** Read real-time vehicle data, including speed and steering angle.
*   **Car Porting:** Comprehensive documentation and community support to add support for your car.
*   **Safety Focused:** Rigorous code review, static analysis, and unit tests ensure safety and reliability.
*   **Extensive Documentation:**  Comprehensive guides and resources for usage, contribution, and extension.
*   **Open Source:**  Contribute and build upon the open-source foundation for advanced vehicle control.

---

opendbc provides a Python API for interacting with your car's internal systems.  This allows you to control features like steering and brakes, as well as reading crucial information like speed. It primarily supports ADAS interfaces, including LKAS and ACC, but aims to read and write as many things as possible (EV charge status, lock/unlocking doors, etc) to build the best vehicle management app ever.

## Getting Started

### Installation & Running Tests
Quickly get started with opendbc using the following commands:

```bash
git clone https://github.com/commaai/opendbc.git
cd opendbc

# Run the automated tests, installation and linting:
./test.sh

# Install dependencies
pip3 install -e .[testing,docs]
# build with 8 cores
scons -j8
# run the tests
pytest .
# run the linter
lefthook run lint
```

### Example Usage

Explore the `examples/` directory for sample programs demonstrating car state reading and control functionalities.
For example, `examples/joystick.py` provides joystick-based car control.

## Project Structure

*   `opendbc/dbc/`: Contains [DBC](https://en.wikipedia.org/wiki/CAN_bus#DBC) files (CAN database files).
*   `opendbc/can/`:  A library for parsing and generating CAN messages from DBC files.
*   `opendbc/car/`:  A high-level library for car interfacing using Python.
*   `opendbc/safety/`:  Provides safety features for supported cars, ensuring secure operation.

## Adding Support for Your Car ("Porting")

Adding support for a new car involves connecting to its CAN bus, reverse-engineering messages, and developing the necessary Python interfaces.
Follow the guides to connect to your car, reverse engineer CAN messages, and understand the structure of a port.

### Structure of a port

Depending on the brand, most of this basic structure will already be in place.

The entirety of a car port lives in `opendbc/car/<brand>/`:
* `carstate.py`: parses out the relevant information from the CAN stream using the car's DBC file
* `carcontroller.py`: outputs CAN messages to control the car
* `<brand>can.py`: thin Python helpers around the DBC file to build CAN messages
* `fingerprints.py`: database of ECU firmware versions for identifying car models
* `interface.py`: high level class for interfacing with the car
* `radar_interface.py`: parses out the radar
* `values.py`: enumerates the brand's supported cars

### Tuning

Use the [longitudinal maneuvers](https://github.com/commaai/openpilot/tree/master/tools/longitudinal_maneuvers) report to evaluate your car's longitudinal control and tune it.

## Contributing

Contributions are welcome! Coordinate with the community on GitHub and [Discord](https://discord.comma.ai) to enhance opendbc. Check out the `#dev-opendbc-cars` channel and `Vehicle Specific` section for collaboration.

### Roadmap

*   [x] `pip install opendbc`
*   [x] 100% type coverage
*   [x] 100% line coverage
*   [x] Make car ports easier: refactors, tools, tests, and docs
*   [x] Expose the state of all supported cars better: https://github.com/commaai/opendbc/issues/1144
*   [ ] Extend support to every car with LKAS + ACC interfaces
*   [ ] Automatic lateral and longitudinal control/tuning evaluation
*   [ ] Auto-tuning for [lateral](https://blog.comma.ai/090release/#torqued-an-auto-tuner-for-lateral-control) and longitudinal control
*   [ ] [Automatic Emergency Braking](https://en.wikipedia.org/wiki/Automated_emergency_braking_system)

## Safety Model

The safety firmware, operating within a [panda](https://comma.ai/shop/panda), defaults to `SAFETY_SILENT` mode. Safety modes optionally support `controls_allowed`, which allows or blocks a subset of messages based on a customizable state in the board.  To utilize the safety features, understanding of the [openpilot safety](https://github.com/commaai/openpilot/blob/master/docs/SAFETY.md) is essential.

## Code Quality & Rigor

opendbc's safety firmware prioritizes code rigor, employing:

*   Static code analysis with [cppcheck](https://github.com/danmar/cppcheck/), including MISRA C:2012 checks.
*   Strict compiler flags: `-Wall -Wextra -Wstrict-prototypes -Werror`.
*   Unit tests with [mutation tests](opendbc/safety/tests/misra/test_mutation.py) and 100% line coverage for the safety unit tests
*   Linters and type checkers like [ruff linter](https://github.com/astral-sh/ruff) and [mypy](https://mypy-lang.org/).

## Bounties

Contribute to opendbc and earn bounties:
*   $2000 - [Any car brand / platform port](https://github.com/orgs/commaai/projects/26/views/1?pane=issue&itemId=47913774)
*   $250 - [Any car model port](https://github.com/orgs/commaai/projects/26/views/1?pane=issue&itemId=47913790)
*   $300 - [Reverse Engineering a new Actuation Message](https://github.com/orgs/commaai/projects/26/views/1?pane=issue&itemId=73445563)

Additional, higher-value bounties are available for popular car models at [comma.ai/bounties](comma.ai/bounties).

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

### More Resources

*   [*How Do We Control The Car?*](https://www.youtube.com/watch?v=nNU6ipme878&pp=ygUoY29tbWEgY29uIDIwMjEgaG93IGRvIHdlIGNvbnRyb2wgdGhlIGNhcg%3D%3D) by [@robbederks](https://github.com/robbederks) from COMMA\_CON 2021
*   [*How to Port a Car*](https://www.youtube.com/watch?v=XxPS5TpTUnI&t=142s&pp=ygUPamFzb24gY29tbWEgY29u) by [@jyoung8607](https://github.com/jyoung8607) from COMMA\_CON 2023
*   [commaCarSegments](https://huggingface.co/datasets/commaai/commaCarSegments): a massive dataset of CAN data from 300 different car models
*   [cabana](https://github.com/commaai/openpilot/tree/master/tools/cabana#readme): our tool for reverse engineering CAN messages
*   [can\_print\_changes.py](https://github.com/commaai/openpilot/blob/master/selfdrive/debug/can_print_changes.py): diff the whole CAN bus across two drives, such as one without any LKAS and one with LKAS
*   [longitudinal maneuvers](https://github.com/commaai/openpilot/tree/master/tools/longitudinal_maneuvers): a tool for evaluating and tuning longitudinal control
*   [opendbc data](https://commaai.github.io/opendbc-data/): a repository of longitudinal maneuver evaluations

## Careers at Comma.ai

Join the team! [comma.ai/jobs](https://comma.ai/jobs) is hiring engineers to work on opendbc and [openpilot](https://github.com/commaai/openpilot). Contribute and grow with us!