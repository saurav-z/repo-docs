<div align="center" style="text-align: center;">

<h1>opendbc: Your Python API for Advanced Vehicle Control</h1>
<p>
  <b>Unlock full control over your vehicle's systems with opendbc, a powerful Python API.</b>
  <br>
  Access and manipulate steering, gas, brakes, and more to build the ultimate vehicle management experience.
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

**opendbc** is a Python API designed to interact with modern vehicles, enabling control over key functions like steering, acceleration, and braking.  This project leverages the latest advancements in automotive technology, specifically those related to Lane Keeping Assist (LKAS) and Adaptive Cruise Control (ACC), to provide a comprehensive interface for vehicle control and data access.  This enables advanced applications for ADAS interfaces for [openpilot](https://github.com/commaai/openpilot) and beyond.

## Key Features

*   **Control**:  Gain control over steering, gas, and brakes in compatible vehicles.
*   **Data Access**: Read real-time data, including speed, steering angle, and more.
*   **Car Compatibility**: Supports a wide range of vehicles with LKAS and ACC systems. (See supported cars list: [docs/CARS.md](docs/CARS.md))
*   **Extensible API**:  Designed to facilitate both reading and writing various vehicle parameters (EV charge status, door locks, etc.).
*   **Community-Driven**:  Open-source and community supported with active development and contributions welcome.  ([Original Repo](https://github.com/commaai/opendbc))

## Quick Start

Get up and running quickly with these simple steps:

```bash
git clone https://github.com/commaai/opendbc.git
cd opendbc

# Run the full suite of dependency installations, compiling, linting, and tests (CI):
./test.sh

# Or, run individual commands:
pip3 install -e .[testing,docs]  # install dependencies
scons -j8                        # build with 8 cores
pytest .                         # run the tests
lefthook run lint                # run the linter
```

Explore the `examples/` directory for sample programs to control your car, including [`examples/joystick.py`](examples/joystick.py), which allows you to control a car with a joystick.

## Project Structure

*   `opendbc/dbc/`: Contains DBC (CAN Database) files.
*   `opendbc/can/`:  A library for parsing and building CAN messages from DBC files.
*   `opendbc/car/`:  A high-level library for Python-based car interfacing.
*   `opendbc/safety/`: Provides functional safety for all supported cars.

## How to Port a Car

This guide details the process of adding support for new cars or improving existing compatibility, including:
*   Connecting to the car with a comma 3X and a car harness.
*   Reversing engineering CAN messages.
*   Car port structure.
*   Tuning longitudinal controls.

The complete guide can be found within the project's documentation, but this section provides a summary.

## Contributing

Contribute to opendbc through GitHub and the Discord community. Check out the `#dev-opendbc-cars` channel and `Vehicle Specific` section for collaboration.

### Roadmap

*   **Short Term:**  Improve installation, enhance code coverage, and simplify car port development.
*   **Longer Term:** Expand car support, and implement auto-tuning capabilities.

## Safety Model

opendbc's safety firmware, designed to work with [openpilot](https://github.com/commaai/openpilot) and [panda](https://github.com/commaai/panda), operates in `SAFETY_SILENT` mode by default.  Specific safety modes can be selected to send messages, and the safety modes optionally support `controls_allowed`, which allows or blocks a subset of messages based on a customizable state in the board.

## Code Rigor

The safety firmware is rigorously tested, including static code analysis, MISRA C:2012 compliance checks, and unit tests with 100% line coverage.

## Bounties

Earn rewards for contributing!  Bounties are available for:

*   Adding support for any car brand/platform: \$2000
*   Adding support for any car model: \$250
*   Reverse Engineering a new Actuation Message: \$300

More details can be found on the [comma.ai/bounties](comma.ai/bounties) page.

## FAQ

*   **How do I use this?** A [comma 3X](https://comma.ai/shop/comma-3x) is the optimal hardware for development and execution of opendbc and openpilot.
*   **Which cars are supported?** Consult the [supported cars list](docs/CARS.md).
*   **Can I add support for my car?** Yes, community contributions are highly encouraged. See the "How to Port a Car" section.
*   **Which cars can be supported?** Vehicles with LKAS and ACC are prime candidates.
*   **How does this work?**  Designed hardware to replace built-in features.
*   **Is there a timeline or roadmap for adding car support?** No set timeline, community contributions are prioritized.

### Terms

*   **port**:  Integration and support for a specific car.
*   **lateral control**:  Steering control.
*   **longitudinal control**: Gas/brake control.
*   **fingerprinting**:  Automated car identification.
*   **LKAS**:  Lane Keeping Assist.
*   **ACC**: Adaptive Cruise Control.
*   **harness**: Car-specific hardware to connect and intercept ADAS messages.
*   **panda**: Hardware used to get on a car's CAN bus.
*   **ECU**: Electronic Control Unit (car's computer).
*   **CAN bus**:  Network connecting ECUs.
*   **cabana**: Tool for reverse engineering CAN messages.
*   **DBC file**:  Defines CAN message formats.
*   **openpilot**: An ADAS system for cars supported by opendbc.
*   **comma**:  The company behind opendbc.
*   **comma 3X**:  Hardware used to run openpilot.

### More Resources

*   [*How Do We Control The Car?*](https://www.youtube.com/watch?v=nNU6ipme878&pp=ygUoY29tbWEgY29uIDIwMjEgaG93IGRvIHdlIGNvbnRyb2wgdGhlIGNhcg%3D%3D) by [@robbederks](https://github.com/robbederks) from COMMA_CON 2021
*   [*How to Port a Car*](https://www.youtube.com/watch?v=XxPS5TpTUnI&t=142s&pp=ygUPamFzb24gY29tbWEgY29u) by [@jyoung8607](https://github.com/jyoung8607) from COMMA_CON 2023
*   [commaCarSegments](https://huggingface.co/datasets/commaai/commaCarSegments): a massive dataset of CAN data from 300 different car models
*   [cabana](https://github.com/commaai/openpilot/tree/master/tools/cabana#readme): our tool for reverse engineering CAN messages
*   [can_print_changes.py](https://github.com/commaai/openpilot/blob/master/selfdrive/debug/can_print_changes.py): diff the whole CAN bus across two drives, such as one without any LKAS and one with LKAS
*   [longitudinal maneuvers](https://github.com/commaai/openpilot/tree/master/tools/longitudinal_maneuvers): a tool for evaluating and tuning longitudinal control
*   [opendbc data](https://commaai.github.io/opendbc-data/): a repository of longitudinal maneuver evaluations

## Join the Team - [comma.ai/jobs](https://comma.ai/jobs)

comma is actively hiring engineers.  We encourage contributions and offer exciting opportunities.