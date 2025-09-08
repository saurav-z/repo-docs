<div align="center" style="text-align: center;">
  <h1>opendbc: Unlock Your Car's Potential with Python</h1>
  <p>
    <b>opendbc provides a powerful Python API to control, read data, and build vehicle management applications for your car, opening the door to advanced automotive features.</b>
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

## Key Features of opendbc:

*   **Control Your Car:** Access and manipulate steering, gas, brakes, and more.
*   **Real-time Data Access:** Read vital information such as speed and steering angle.
*   **Open Source & Community Driven:** Built with community contributions, enabling extensive car support.
*   **ADAS Integration:** Designed for openpilot, but applicable to many vehicle management applications.

🔗 [Visit the opendbc GitHub Repository](https://github.com/commaai/opendbc) for more information.

---

opendbc is a Python API designed to interact with your car's internal systems. It allows you to control and read data from your car's electronic systems, expanding the possibilities for automotive development. This project's primary goal is to support ADAS interfaces, particularly for openpilot, while offering the tools needed to read and write various data points like EV charge status and door lock/unlock controls.

## Getting Started

### Installation

```bash
git clone https://github.com/commaai/opendbc.git
cd opendbc

# Comprehensive installation, building, linting, and testing.
./test.sh

# Individual commands
pip3 install -e .[testing,docs]  # install dependencies
scons -j8                        # build with 8 cores
pytest .                         # run the tests
lefthook run lint                # run the linter
```

### Examples

Explore the `examples/` directory for sample programs. The `examples/joystick.py` script provides a simple way to control your car using a joystick.

### Project Structure

*   `opendbc/dbc/`: Stores [DBC](https://en.wikipedia.org/wiki/CAN_bus#DBC_(CAN_Database_Files)) files.
*   `opendbc/can/`: Provides a library for parsing and building CAN messages from DBC files.
*   `opendbc/car/`: Contains a high-level Python library for car interfacing.
*   `opendbc/safety/`: Implements functional safety mechanisms for supported cars.

## How to Port a Car

Expand your car's support by adding a car port, integrating features to control steering, gas, and brakes. A complete car port includes lateral control, longitudinal control, tuning, radar parsing, fingerprinting, and more.

### Steps for Car Integration

1.  **Connect to the Car:** Use a comma 3X and car harness (available at comma.ai/shop) to connect to the CAN buses.
2.  **Port Structure:** A car port resides in `opendbc/car/<brand>/`.
    *   `carstate.py`: Parses data from the CAN stream.
    *   `carcontroller.py`: Sends control messages.
    *   `<brand>can.py`: Helper functions for building CAN messages.
    *   `fingerprints.py`: ECU firmware identification.
    *   `interface.py`: Car interface class.
    *   `radar_interface.py`: Parses radar data.
    *   `values.py`: Car support enumeration.
3.  **Reverse Engineer CAN Messages:** Use [cabana](https://github.com/commaai/openpilot/tree/master/tools/cabana) to analyze CAN data.
4.  **Tuning:** Evaluate and tune longitudinal control using the [longitudinal maneuvers](https://github.com/commaai/openpilot/tree/master/tools/longitudinal_maneuvers) tool.

## Contributing

Join the opendbc community on GitHub and [Discord](https://discord.comma.ai) to contribute. Engage in the `#dev-opendbc-cars` channel and `Vehicle Specific` section to get started.

### Roadmap

**Short Term:**

*   `pip install opendbc`
*   100% type and line coverage.
*   Improve car port ease: refactors, tools, tests, and docs.
*   Improve car state visibility: ([#1144](https://github.com/commaai/opendbc/issues/1144))

**Longer Term:**

*   Expand support to every car with LKAS + ACC.
*   Automate lateral and longitudinal control/tuning evaluation.
*   Implement auto-tuning for [lateral](https://blog.comma.ai/090release/#torqued-an-auto-tuner-for-lateral-control) and longitudinal control.
*   Add [Automatic Emergency Braking](https://en.wikipedia.org/wiki/Automated_emergency_braking_system).

Contributions are welcome.

## Safety Model

The `opendbc/safety` directory, in conjunction with the openpilot safety model, provides essential safety features for the Panda. Safety modes, such as `SAFETY_SILENT` and `SAFETY_ALLOUTPUT`, ensure the integrity of CAN bus interactions.

### Code Rigor

*   Static code analysis using [cppcheck](https://github.com/danmar/cppcheck/) and [MISRA C:2012](https://misra.org.uk/) checks.
*   Strict compiler flags (`-Wall -Wextra -Wstrict-prototypes -Werror`).
*   Unit tests for all car variants.
*   Mutation and 100% line coverage testing.
*   Ruff linter and mypy checks for the car interface library.

### Bounties

Participate in the opendbc project and earn rewards.

*   $2000 - [Any car brand / platform port](https://github.com/orgs/commaai/projects/26/views/1?pane=issue&itemId=47913774)
*   $250 - [Any car model port](https://github.com/orgs/commaai/projects/26/views/1?pane=issue&itemId=47913790)
*   $300 - [Reverse Engineering a new Actuation Message](https://github.com/orgs/commaai/projects/26/views/1?pane=issue&itemId=73445563)

Discover additional bounties for popular cars at [comma.ai/bounties](comma.ai/bounties).

## FAQ

*   **How do I use this?** A [comma 3X](https://comma.ai/shop/comma-3x) is best for running opendbc and openpilot.
*   **Which cars are supported?** Check the [supported cars list](docs/CARS.md).
*   **Can I add support for my car?** Yes, see the [car porting guide](https://github.com/commaai/opendbc/blob/docs/README.md#how-to-port-a-car).
*   **Which cars can be supported?** Cars with LKAS and ACC features. Additional info [here](https://github.com/commaai/openpilot/blob/master/docs/CARS.md#dont-see-your-car-here).
*   **How does this work?** Hardware designed to replace built-in lane keep and cruise features. See [this talk](https://www.youtube.com/watch?v=FL8CxUSfipM).
*   **Is there a timeline for adding car support?** Community-driven, with comma validating.

### Terms

*   **port**: Integrating a specific car.
*   **lateral control**: Steering control.
*   **longitudinal control**: Gas/brakes control.
*   **fingerprinting**: Car identification.
*   **[LKAS](https://en.wikipedia.org/wiki/Lane_departure_warning_system)**: Lane keeping assist.
*   **[ACC](https://en.wikipedia.org/wiki/Adaptive_cruise_control)**: Adaptive cruise control.
*   **[harness](https://comma.ai/shop/car-harness)**: Connectors.
*   **[panda](https://github.com/commaai/panda)**: CAN bus hardware.
*   **[ECU](https://en.wikipedia.org/wiki/Electronic_control_unit)**: Car control module.
*   **[CAN bus](https://en.wikipedia.org/wiki/CAN_bus)**: Car network.
*   **[cabana](https://github.com/commaai/openpilot/tree/master/tools/cabana#readme)**: CAN message tool.
*   **[DBC file](https://en.wikipedia.org/wiki/CAN_bus#DBC)**: CAN definitions.
*   **[openpilot](https://github.com/commaai/openpilot)**: ADAS software.
*   **[comma](https://github.com/commaai)**: The company.
*   **[comma 3X](https://comma.ai/shop/comma-3x)**: Hardware to run openpilot.

### More resources

*   [*How Do We Control The Car?*](https://www.youtube.com/watch?v=nNU6ipme878&pp=ygUoY29tbWEgY29uIDIwMjEgaG93IGRvIHdlIGNvbnRyb2wgdGhlIGNhcg%3D%3D) by [@robbederks](https://github.com/robbederks) from COMMA_CON 2021
*   [*How to Port a Car*](https://www.youtube.com/watch?v=XxPS5TpTUnI&t=142s&pp=ygUPamFzb24gY29tbWEgY29u) by [@jyoung8607](https://github.com/jyoung8607) from COMMA_CON 2023
*   [commaCarSegments](https://huggingface.co/datasets/commaai/commaCarSegments): A massive dataset of CAN data from 300 different car models
*   [cabana](https://github.com/commaai/openpilot/tree/master/tools/cabana#readme): Our tool for reverse engineering CAN messages
*   [can_print_changes.py](https://github.com/commaai/openpilot/blob/master/selfdrive/debug/can_print_changes.py): Diff the whole CAN bus across two drives, such as one without any LKAS and one with LKAS
*   [longitudinal maneuvers](https://github.com/commaai/openpilot/tree/master/tools/longitudinal_maneuvers): A tool for evaluating and tuning longitudinal control
*   [opendbc data](https://commaai.github.io/opendbc-data/): a repository of longitudinal maneuver evaluations

## Join the Team - [comma.ai/jobs](https://comma.ai/jobs)

comma is hiring engineers to work on opendbc and [openpilot](https://github.com/commaai/openpilot). Contribute today!