<div align="center">
  <img src="https://img.shields.io/github/stars/commaai/opendbc?style=social" alt="GitHub stars">
  <h1>opendbc: Open Source Car Control API</h1>
  <p><b>Control and access your car's data with opendbc, a powerful Python API.</b></p>
  <p>Take control of your car's steering, gas, brakes, and access vital data such as speed and steering angle with this open-source project.</p>

  <a href="https://github.com/commaai/opendbc">
    <img src="https://img.shields.io/badge/GitHub-View%20on%20GitHub-blue?style=flat&logo=github" alt="GitHub">
  </a>
  <a href="https://docs.comma.ai">
    <img src="https://img.shields.io/badge/Docs-Read%20the%20docs-brightgreen?style=flat&logo=readthedocs" alt="Docs">
  </a>
  <a href="https://discord.comma.ai">
    <img src="https://img.shields.io/discord/469524606043160576?logo=discord&label=Discord&color=7289DA" alt="Discord">
  </a>
</div>

---

## Key Features

*   **Control:** Command steering, gas, and brakes.
*   **Read Data:** Access car data like speed and steering angle.
*   **Open Source:**  Leverage a community-driven project.
*   **Extensive Support:**  Supports a growing list of car makes and models (see [supported cars list](docs/CARS.md)).
*   **Python API:** Easy to use Python interface for developers.
*   **ADAS Integration:**  Designed to interface with ADAS systems like openpilot.
*   **Community Driven:** Actively developed and maintained with contributions from the community.

---

opendbc is a Python API that gives you control of your car. It's designed to work with cars that have electronically-actuated steering, gas, and brakes, such as those equipped with LKAS and ACC. This project allows you to control the steering, gas, and brakes on many vehicles, while also providing data like speed and steering angle.

## Quickstart Guide

Get started with opendbc using the following steps:

```bash
git clone https://github.com/commaai/opendbc.git
cd opendbc

# Install dependencies, compile, run linting, and tests
./test.sh

# Individual commands executed by test.sh
pip3 install -e .[testing,docs]  # Install dependencies
scons -j8                        # Build with 8 cores
pytest .                         # Run tests
lefthook run lint                # Run linter
```

Explore example programs within the [`examples/`](examples/) directory to get a feel for controlling your car's features, and even control your car with a joystick using [`examples/joystick.py`](examples/joystick.py).

### Project Structure

*   [`opendbc/dbc/`](opendbc/dbc/) -  Houses [DBC](https://en.wikipedia.org/wiki/CAN_bus#DBC) files, crucial for car communication.
*   [`opendbc/can/`](opendbc/can/) - A library for parsing and constructing CAN messages from DBC files.
*   [`opendbc/car/`](opendbc/car/) - A high-level Python library for interacting with vehicles.
*   [`opendbc/safety/`](opendbc/safety/) -  Manages the functional safety aspects for vehicles supported by `opendbc/car/`.

## How to Port a Car

This guide provides a comprehensive walkthrough for adding support to a new car. Start by connecting to the car with a comma 3X and a car harness, then reverse engineer CAN messages using tools like cabana, and tune your car's longitudinal controls.

## Contributing

Contribute to opendbc on GitHub and the [Discord server](https://discord.comma.ai), specifically in the `#dev-opendbc-cars` channel.

### Roadmap

*   **Short Term Goals:**
    *   `pip install opendbc`
    *   100% type and line coverage
    *   Improve car port ease with refactors, tools, tests, and documentation.
    *   Better state exposition for all supported cars.

*   **Longer Term Goals:**
    *   Expand support to every car with LKAS and ACC interfaces.
    *   Automated lateral and longitudinal control and tuning evaluation.
    *   Implement Auto-tuning for lateral and longitudinal control.
    *   [Automatic Emergency Braking](https://en.wikipedia.org/wiki/Automated_emergency_braking_system).

## Safety Model

opendbc's safety firmware, when used with a [panda](https://comma.ai/shop/panda), defaults to `SAFETY_SILENT` mode for CAN bus security. Safety modes, like `SAFETY_ALLOUTPUT`, allow message transmission based on customizable board states. The [CI regression tests](https://github.com/commaai/opendbc/actions) enforce rigorous code quality and safety standards.

## Code Rigor

The opendbc safety firmware is written with rigorous application code standards to provide a high level of functional safety. These standards include:

*   Static code analysis with [cppcheck](https://github.com/danmar/cppcheck/)
*   [MISRA C:2012](https://misra.org.uk/) violation checks using cppcheck
*   Strict compiler options including flags `-Wall -Wextra -Wstrict-prototypes -Werror`
*   Unit tests using [unit tests](opendbc/safety/tests) for each supported car variant
*   Mutation and line coverage tests

## Bounties

Contribute and get paid! Earn bounties for adding car support.

*   $2000 - [Any car brand / platform port](https://github.com/orgs/commaai/projects/26/views/1?pane=issue&itemId=47913774)
*   $250 - [Any car model port](https://github.com/orgs/commaai/projects/26/views/1?pane=issue&itemId=47913790)
*   $300 - [Reverse Engineering a new Actuation Message](https://github.com/orgs/commaai/projects/26/views/1?pane=issue&itemId=73445563)

Find more bounty opportunities at [comma.ai/bounties](comma.ai/bounties).

## FAQ

*   **How do I use this?** A [comma 3X](https://comma.ai/shop/comma-3x) is custom-designed to be the best way to run and develop opendbc and openpilot.
*   **Which cars are supported?** See the [supported cars list](docs/CARS.md).
*   **Can I add support for my car?** Yes, most car support comes from the community. Read the guide [here](https://github.com/commaai/opendbc/blob/docs/README.md#how-to-port-a-car).
*   **Which cars can be supported?** Any car with LKAS and ACC. More info [here](https://github.com/commaai/openpilot/blob/master/docs/CARS.md#dont-see-your-car-here).
*   **How does this work?** In short, we designed hardware to replace your car's built-in lane keep and adaptive cruise features. See [this talk](https://www.youtube.com/watch?v=FL8CxUSfipM) for an in-depth explanation.
*   **Is there a timeline or roadmap for adding car support?** No, most car support comes from the community, with comma doing final safety and quality validation. The more complete the community car port is and the more popular the car is, the more likely we are to pick it up as the next one to validate.

### Terms

*   **port**: Car integration and support
*   **lateral control**: Steering control
*   **longitudinal control**: Gas/brakes control
*   **fingerprinting**: Automatic car identification.
*   **[LKAS](https://en.wikipedia.org/wiki/Lane_departure_warning_system)**: Lane Keeping Assist
*   **[ACC](https://en.wikipedia.org/wiki/Adaptive_cruise_control)**: Adaptive Cruise Control
*   **[harness](https://comma.ai/shop/car-harness)**: Car-specific hardware connection
*   **[panda](https://github.com/commaai/panda)**: CAN bus access hardware
*   **[ECU](https://en.wikipedia.org/wiki/Electronic_control_unit)**: Car's control modules
*   **[CAN bus](https://en.wikipedia.org/wiki/CAN_bus)**: Communication bus for ECUs
*   **[cabana](https://github.com/commaai/openpilot/tree/master/tools/cabana#readme)**: CAN message reverse engineering tool
*   **[DBC file](https://en.wikipedia.org/wiki/CAN_bus#DBC)**: Defines CAN messages
*   **[openpilot](https://github.com/commaai/openpilot)**: ADAS system using opendbc
*   **[comma](https://github.com/commaai)**: opendbc's creator
*   **[comma 3X](https://comma.ai/shop/comma-3x)**: Hardware for opendbc and openpilot

### More Resources

*   [*How Do We Control The Car?*](https://www.youtube.com/watch?v=nNU6ipme878&pp=ygUoY29tbWEgY29uIDIwMjEgaG93IGRvIHdlIGNvbnRyb2wgdGhlIGNhcg%3D%3D)
*   [*How to Port a Car*](https://www.youtube.com/watch?v=XxPS5TpTUnI&t=142s&pp=ygUPamFzb24gY29tbWEgY29u)
*   [commaCarSegments](https://huggingface.co/datasets/commaai/commaCarSegments)
*   [cabana](https://github.com/commaai/openpilot/tree/master/tools/cabana#readme)
*   [can_print_changes.py](https://github.com/commaai/openpilot/blob/master/selfdrive/debug/can_print_changes.py)
*   [longitudinal maneuvers](https://github.com/commaai/openpilot/tree/master/tools/longitudinal_maneuvers)
*   [opendbc data](https://commaai.github.io/opendbc-data/)

---

## Join the Team!

[comma.ai](https://comma.ai/jobs) is hiring! Work on opendbc and [openpilot](https://github.com/commaai/openpilot) and join a team that values community contributions.