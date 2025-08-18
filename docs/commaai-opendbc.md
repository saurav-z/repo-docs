<div align="center">
  <a href="https://github.com/commaai/opendbc">
    <img src="https://img.shields.io/github/stars/commaai/opendbc?style=social" alt="GitHub stars">
  </a>
  <h1>opendbc: Open-Source Car Control and Data Access</h1>
  <p><b>opendbc provides a powerful Python API to control your car's critical functions and access real-time data.</b></p>
  <p>
    <a href="https://docs.comma.ai">Documentation</a>
    <span> · </span>
    <a href="https://github.com/commaai/openpilot/blob/master/docs/CONTRIBUTING.md">Contribute</a>
    <span> · </span>
    <a href="https://discord.comma.ai">Discord</a>
  </p>
  <p>
    <a href="https://x.com/comma_ai">
      <img alt="Twitter Follow" src="https://img.shields.io/twitter/follow/comma_ai?style=social"/>
    </a>
    <a href="https://discord.comma.ai">
      <img alt="Discord" src="https://img.shields.io/discord/469524606043160576"/>
    </a>
    <a href="LICENSE">
      <img alt="License" src="https://img.shields.io/badge/License-MIT-yellow.svg"/>
    </a>
  </p>
</div>

---

opendbc is a Python API designed to interface with your car's internal systems, offering control and data access for a wide range of vehicles.  It allows developers to control steering, gas, brakes, and more, while also reading crucial data such as speed and steering angle.  Primarily developed to support ADAS interfaces for [openpilot](https://github.com/commaai/openpilot), opendbc aims to create the ultimate vehicle management application.

Key Features:

*   **Control:** Actuate steering, gas, and brakes.
*   **Data Access:** Read real-time vehicle data like speed and steering angle.
*   **Cross-Platform:**  Developed in Python, supporting a wide range of systems.
*   **Open Source:**  Contribute to the project and extend support for more cars.
*   **Community Driven:** Collaborate with the community on GitHub and Discord.

---

## Getting Started

### Installation

```bash
git clone https://github.com/commaai/opendbc.git
cd opendbc

# Install dependencies and run tests
./test.sh

# Individual Commands (for advanced users)
pip3 install -e .[testing,docs]  # install dependencies
scons -j8                        # build with 8 cores
pytest .                         # run the tests
lefthook run lint                # run the linter
```

### Examples

Explore the `examples/` directory for ready-to-use code.  `examples/joystick.py` offers real-time car control with a joystick.

## Project Structure

*   [`opendbc/dbc/`](opendbc/dbc/): Contains [DBC](https://en.wikipedia.org/wiki/CAN_bus#DBC) files (CAN bus database descriptions).
*   [`opendbc/can/`](opendbc/can/):  A library to parse and construct CAN messages.
*   [`opendbc/car/`](opendbc/car/):  A high-level Python library to interact with vehicles.
*   [`opendbc/safety/`](opendbc/safety/): Safety mechanisms for supported cars.

## How to Add Car Support

This guide covers the complete process, from adding new cars to enhancing existing ones.

### Hardware Setup

1.  Connect to your car using a [comma 3X](https://comma.ai/shop/comma-3x) and a car harness.
2.  If a compatible harness isn't available, use a developer harness from comma.ai/shop and connect to the appropriate CAN buses.

### Car Port Structure

A car port is housed within `opendbc/car/<brand>/` and includes:

*   `carstate.py`:  Parses vehicle data from CAN streams.
*   `carcontroller.py`: Generates CAN messages for car control.
*   `<brand>can.py`: Helper functions for building CAN messages.
*   `fingerprints.py`:  Database of ECU firmware versions.
*   `interface.py`: High-level interface class for car interaction.
*   `radar_interface.py`:  For parsing radar data.
*   `values.py`: Defines the brand's supported cars.

### Reverse Engineering CAN Messages

1.  Record a drive with various events (LKAS, ACC, steering).
2.  Analyze the recording in [cabana](https://github.com/commaai/openpilot/tree/master/tools/cabana) to understand CAN message structures.

### Tuning

Use the [longitudinal maneuvers](https://github.com/commaai/openpilot/tree/master/tools/longitudinal_maneuvers) report to evaluate and tune longitudinal control.

## Contributing

Development is coordinated on GitHub and [Discord](https://discord.comma.ai) (channel `#dev-opendbc-cars`).

### Roadmap

Short Term:

*   \[ ] `pip install opendbc`
*   \[ ] 100% type coverage
*   \[ ] 100% line coverage
*   \[ ] Improve car port development: refactors, tools, tests, and documentation
*   \[ ] Enhance the state of supported cars: [Issue 1144](https://github.com/commaai/opendbc/issues/1144)

Longer Term:

*   \[ ] Expand support to all cars with LKAS and ACC.
*   \[ ] Automatic lateral and longitudinal control/tuning evaluation.
*   \[ ] Implement auto-tuning for lateral and longitudinal control.
*   \[ ] Develop Automatic Emergency Braking.

Contributions are welcome!

## Safety Model

The opendbc safety firmware, used with openpilot and [panda](https://comma.ai/shop/panda), operates in `SAFETY_SILENT` mode by default.  Custom builds can enable additional safety modes.

### Code Quality

The `safety` folder is held to high standards:

*   Static code analysis with [cppcheck](https://github.com/danmar/cppcheck/), including [MISRA C:2012](https://misra.org.uk/) checks.
*   Strict compiler flags (`-Wall -Wextra -Wstrict-prototypes -Werror`).
*   Unit tests for safety logic (`opendbc/safety/tests`).
*   Mutation testing on MISRA coverage.
*   100% line coverage enforced on the safety unit tests.
*   Ruff linter and mypy for the car interface library.

### Bounties

Earn bounties for your contributions!

*   $2000 - [Any car brand / platform port](https://github.com/orgs/commaai/projects/26/views/1?pane=issue&itemId=47913774)
*   $250 - [Any car model port](https://github.com/orgs/commaai/projects/26/views/1?pane=issue&itemId=47913790)
*   $300 - [Reverse Engineering a new Actuation Message](https://github.com/orgs/commaai/projects/26/views/1?pane=issue&itemId=73445563)

See [comma.ai/bounties](comma.ai/bounties) for bounties on popular cars.

## FAQ

*   **How do I use this?**  Use a [comma 3X](https://comma.ai/shop/comma-3x)
*   **Which cars are supported?**  See the [supported cars list](docs/CARS.md).
*   **Can I add support for my car?**  Yes, the community drives most car support.  See the guide in the "How to Port a Car" section.
*   **Which cars can be supported?** Cars with LKAS and ACC. More info [here](https://github.com/commaai/openpilot/blob/master/docs/CARS.md#dont-see-your-car-here).
*   **How does this work?**  Hardware is designed to replace your car's lane keep and adaptive cruise features.
*   **Is there a timeline or roadmap for adding car support?**  No. The community drives car support.

### Glossary

*   **port**: Integration and support for a car.
*   **lateral control**: Steering control.
*   **longitudinal control**: Gas/brake control.
*   **fingerprinting**: Automatic car identification.
*   **[LKAS](https://en.wikipedia.org/wiki/Lane_departure_warning_system)**: Lane keeping assist.
*   **[ACC](https://en.wikipedia.org/wiki/Adaptive_cruise_control)**: Adaptive cruise control.
*   **[harness](https://comma.ai/shop/car-harness)**: Car-specific hardware for ADAS message interception.
*   **[panda](https://github.com/commaai/panda)**: CAN bus access hardware.
*   **[ECU](https://en.wikipedia.org/wiki/Electronic_control_unit)**: Car control modules.
*   **[CAN bus](https://en.wikipedia.org/wiki/CAN_bus)**: ECU communication bus.
*   **[cabana](https://github.com/commaai/openpilot/tree/master/tools/cabana#readme)**: CAN message reverse engineering tool.
*   **[DBC file](https://en.wikipedia.org/wiki/CAN_bus#DBC)**: CAN bus message definitions.
*   **[openpilot](https://github.com/commaai/openpilot)**: ADAS system using opendbc.
*   **[comma](https://github.com/commaai)**: Company behind opendbc.
*   **[comma 3X](https://comma.ai/shop/comma-3x)**: Hardware for running openpilot.

### More Resources

*   [*How Do We Control The Car?*](https://www.youtube.com/watch?v=nNU6ipme878&pp=ygUoY29tbWEgY29uIDIwMjEgaG93IGRvIHdlIGNvbnRyb2wgdGhlIGNhcg%3D%3D)
*   [*How to Port a Car*](https://www.youtube.com/watch?v=XxPS5TpTUnI&t=142s&pp=ygUPamFzb24gY29tbWEgY29u)
*   [commaCarSegments](https://huggingface.co/datasets/commaai/commaCarSegments)
*   [cabana](https://github.com/commaai/openpilot/tree/master/tools/cabana#readme)
*   [can\_print\_changes.py](https://github.com/commaai/openpilot/blob/master/selfdrive/debug/can_print_changes.py)
*   [longitudinal maneuvers](https://github.com/commaai/openpilot/tree/master/tools/longitudinal_maneuvers)
*   [opendbc data](https://commaai.github.io/opendbc-data/)

## Join Our Team - [comma.ai/jobs](https://comma.ai/jobs)

comma is actively recruiting engineers to work on opendbc and [openpilot](https://github.com/commaai/openpilot).  We value community contributions.