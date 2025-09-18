<div align="center" style="text-align: center;">

<h1>opendbc: Your Python API for Car Control and Data</h1>
<p>
  <b>Unlock your car's potential! opendbc empowers you to control steering, gas, brakes, and access vital data, transforming your vehicle into a programmable platform.</b>
  <br>
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

opendbc provides a powerful Python API for interacting with your car's internal systems.  Leveraging the capabilities of [LKAS](https://en.wikipedia.org/wiki/Lane_departure_warning_system#Lane_keeping_and_next_technologies) and [ACC](https://en.wikipedia.org/wiki/Adaptive_cruise_control), opendbc allows you to control steering, gas, and brakes and read critical vehicle data like speed and steering angle.  It's designed to support ADAS interfaces, particularly for [openpilot](https://github.com/commaai/openpilot), but also enables a wide range of vehicle management applications.

**Key Features:**

*   **Control:**  Precise control over steering, acceleration, and braking systems.
*   **Data Acquisition:** Real-time access to crucial vehicle data (speed, steering angle, etc.).
*   **Extensibility:** Supports reading and writing various car signals like EV charge status and door locks.
*   **Open Source:**  Contribute to and enhance the capabilities of opendbc.
*   **Cross-Platform:** Built on Python, it can work on various hardware with a car's CAN Bus.

[Visit the original repository for complete details and contributions](https://github.com/commaai/opendbc).

## Getting Started

Quickly set up and start using opendbc:

```bash
git clone https://github.com/commaai/opendbc.git
cd opendbc

# This command installs dependencies, builds, lints, and runs tests.
./test.sh

# Alternatively, run individual commands:
pip3 install -e .[testing,docs]  # install dependencies
scons -j8                        # build with 8 cores
pytest .                         # run the tests
lefthook run lint                # run the linter
```

Explore example programs in the [`examples/`](examples/) directory, including [`examples/joystick.py`](examples/joystick.py) to control a car with a joystick.

### Project Structure
* [`opendbc/dbc/`](opendbc/dbc/) is a repository of [DBC](https://en.wikipedia.org/wiki/CAN_bus#DBC_(CAN_Database_Files)) files
* [`opendbc/can/`](opendbc/can/) is a library for parsing and building CAN messages from DBC files
* [`opendbc/car/`](opendbc/car/) is a high-level library for interfacing with cars using Python
* [`opendbc/safety/`](opendbc/safety/) is the functional safety for all the cars supported by `opendbc/car/`

## Car Porting Guide

Extend opendbc to support your car! This comprehensive guide covers everything from initial setup to advanced features.

### Connect to the Car

Connect to the car using a comma 3X and a car harness.  Harnesses are available at comma.ai/shop, or you can create your own using a developer harness.

### Structure of a Port

Each car port resides within `opendbc/car/<brand>/`, containing:

*   `carstate.py`: CAN data parsing
*   `carcontroller.py`: Control message output
*   `<brand>can.py`: CAN message helpers
*   `fingerprints.py`: ECU firmware identification
*   `interface.py`: High-level car interface
*   `radar_interface.py`: Radar parsing
*   `values.py`: Car-specific data definitions

### Reverse Engineering CAN Messages

Record driving data, load it into [cabana](https://github.com/commaai/openpilot/tree/master/tools/cabana), and analyze CAN message data.

### Tuning

Evaluate and optimize longitudinal control using the [longitudinal maneuvers](https://github.com/commaai/openpilot/tree/master/tools/longitudinal_maneuvers) report.

## Contributing

Join the opendbc community!  Coordinate development on GitHub and [Discord](https://discord.comma.ai) in the `#dev-opendbc-cars` channel.

### Roadmap

*   [ ] `pip install opendbc`
*   [ ] 100% type coverage
*   [ ] 100% line coverage
*   [ ] Improve car porting tools and documentation.
*   [ ] Enhance car state information: https://github.com/commaai/opendbc/issues/1144
*   [ ] Expand support to all LKAS/ACC-equipped cars.
*   [ ] Automate lateral and longitudinal control evaluation.
*   [ ] Implement auto-tuning for lateral and longitudinal control.
*   [ ] Develop Automatic Emergency Braking features.

Your contributions are welcome!

## Safety Model

opendbc's safety firmware, designed for use with [openpilot](https://github.com/commaai/openpilot) and [panda](https://github.com/commaai/panda), enforces safety through a strict safety model, including silent mode and selectable safety modes.

### Code Rigor

The `safety` folder emphasizes high code quality:

*   Static analysis with [cppcheck](https://github.com/danmar/cppcheck/), including MISRA C:2012 checks.
*   Strict compiler flags (`-Wall -Wextra -Wstrict-prototypes -Werror`).
*   Unit tests ([opendbc/safety/tests](opendbc/safety/tests)) with 100% line coverage.
*   Mutation testing on MISRA coverage.
*   [ruff linter](https://github.com/astral-sh/ruff) and [mypy](https://mypy-lang.org/) for car interface library.

### Bounties

Contribute and earn bounties!

*   $2000 - [Any car brand / platform port](https://github.com/orgs/commaai/projects/26/views/1?pane=issue&itemId=47913774)
*   $250 - [Any car model port](https://github.com/orgs/commaai/projects/26/views/1?pane=issue&itemId=47913790)
*   $300 - [Reverse Engineering a new Actuation Message](https://github.com/orgs/commaai/projects/26/views/1?pane=issue&itemId=73445563)

Higher value bounties are also available for popular cars, as listed at [comma.ai/bounties](comma.ai/bounties).

## FAQ

*   **How do I use this?** A [comma 3X](https://comma.ai/shop/comma-3x) is recommended for development.
*   **Which cars are supported?** See the [supported cars list](docs/CARS.md).
*   **Can I add support for my car?** Yes! See the [car porting guide](https://github.com/commaai/opendbc/blob/docs/README.md#how-to-port-a-car).
*   **Which cars can be supported?**  Cars with LKAS and ACC. More info [here](https://github.com/commaai/openpilot/blob/master/docs/CARS.md#dont-see-your-car-here).
*   **How does this work?** We replace your car's lane keep and cruise features with our hardware. See [this talk](https://www.youtube.com/watch?v=FL8CxUSfipM).
*   **Is there a timeline for adding car support?** Community-driven, with comma validating popular ports.

### Terms

*   **port**: Car integration
*   **lateral control**: Steering control
*   **longitudinal control**: Gas/brakes control
*   **fingerprinting**: Car model identification
*   **[LKAS](https://en.wikipedia.org/wiki/Lane_departure_warning_system)**: Lane keeping
*   **[ACC](https://en.wikipedia.org/wiki/Adaptive_cruise_control)**: Adaptive cruise control
*   **[harness](https://comma.ai/shop/car-harness)**: Car-specific hardware
*   **[panda](https://github.com/commaai/panda)**: CAN bus hardware
*   **[ECU](https://en.wikipedia.org/wiki/Electronic_control_unit)**: Car's computers
*   **[CAN bus](https://en.wikipedia.org/wiki/CAN_bus)**: Inter-ECU communication
*   **[cabana](https://github.com/commaai/openpilot/tree/master/tools/cabana#readme)**: CAN message analysis tool
*   **[DBC file](https://en.wikipedia.org/wiki/CAN_bus#DBC)**: CAN message definitions
*   **[openpilot](https://github.com/commaai/openpilot)**: ADAS system using opendbc
*   **[comma](https://github.com/commaai)**: The company behind opendbc
*   **[comma 3X](https://comma.ai/shop/comma-3x)**: Hardware for running openpilot

### Resources

*   [*How Do We Control The Car?*](https://www.youtube.com/watch?v=nNU6ipme878&pp=ygUoY29tbWEgY29uIDIwMjEgaG93IGRvIHdlIGNvbnRyb2wgdGhlIGNhcg%3D%3D) by [@robbederks](https://github.com/robbederks)
*   [*How to Port a Car*](https://www.youtube.com/watch?v=XxPS5TpTUnI&t=142s&pp=ygUPamFzb24gY29tbWEgY29u) by [@jyoung8607](https://github.com/jyoung8607)
*   [commaCarSegments](https://huggingface.co/datasets/commaai/commaCarSegments): CAN data dataset
*   [cabana](https://github.com/commaai/openpilot/tree/master/tools/cabana#readme): CAN message analysis
*   [can_print_changes.py](https://github.com/commaai/openpilot/blob/master/selfdrive/debug/can_print_changes.py): CAN bus diff tool
*   [longitudinal maneuvers](https://github.com/commaai/openpilot/tree/master/tools/longitudinal_maneuvers): Longitudinal control evaluation
*   [opendbc data](https://commaai.github.io/opendbc-data/): Longitudinal maneuver data

## Join the Team - [comma.ai/jobs](https://comma.ai/jobs)

comma is hiring!  We welcome contributions.