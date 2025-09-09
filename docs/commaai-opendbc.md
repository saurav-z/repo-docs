<div align="center" style="text-align: center;">

<h1>opendbc: Open Source Car Control API</h1>
<p>
  <b>Take control of your car with opendbc, a powerful Python API.</b>  Unlock the ability to read and write to your car's systems, including steering, gas, brakes, and more.
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

**opendbc** is a powerful Python API designed to give you unprecedented control over your vehicle. This project aims to provide a comprehensive interface for reading and writing data to a wide range of modern vehicles, enabling control of steering, gas, brakes, and more. Primarily supporting ADAS interfaces for [openpilot](https://github.com/commaai/openpilot), opendbc also allows users to interact with other vehicle systems like EV charging status and door locks.

**[View the opendbc repository on GitHub](https://github.com/commaai/opendbc)**

**Key Features:**

*   **Car Control:** Interface with steering, gas, and brakes.
*   **Data Acquisition:** Read vital vehicle data, including speed and steering angle.
*   **Extensive Car Support:** Focused on supporting cars with LKAS and ACC systems.
*   **Community Driven:** Benefit from a thriving community and active development.
*   **Flexible Applications:** Build vehicle management apps, custom controls, and more.

---

The primary documentation for the opendbc project is within this README and the [supported cars list](docs/CARS.md).

## Getting Started

Quickly get up and running with these commands:

```bash
git clone https://github.com/commaai/opendbc.git
cd opendbc

# Installs dependencies, builds, lints, and runs tests.
./test.sh

# Individual commands for reference
pip3 install -e .[testing,docs]  # install dependencies
scons -j8                        # build with 8 cores
pytest .                         # run the tests
lefthook run lint                # run the linter
```

Explore the [`examples/`](examples/) directory for sample programs to read car data and control systems, and [`examples/joystick.py`](examples/joystick.py) which controls a car with a joystick.

### Project Structure

*   [`opendbc/dbc/`](opendbc/dbc/) - Contains DBC (CAN Database) files for supported vehicles.
*   [`opendbc/can/`](opendbc/can/) - A library for parsing and building CAN messages from DBC files.
*   [`opendbc/car/`](opendbc/car/) - High-level Python library for interacting with cars.
*   [`opendbc/safety/`](opendbc/safety/) - Provides functional safety features for the cars supported by `opendbc/car/`.

## How to Port a Car

Expanding opendbc to support new vehicles is a community-driven process. This section offers a detailed guide for contributing car support, covering everything from adding new car models to refining existing integrations. The new car support docs will clearly communicate each car's support level.

### Connect to the Car

Begin by connecting to your car using a comma 3X and a car harness. Harnesses connect to two CAN buses, one of which is split to send actuation messages.

Find harnesses at comma.ai/shop, or start with a "developer harness" and create your own custom connector.

### Structure of a Port

A typical car port structure, located in `opendbc/car/<brand>/`, consists of:

*   `carstate.py`: Parses data from the CAN stream.
*   `carcontroller.py`: Outputs CAN messages for control.
*   `<brand>can.py`: Python helpers for building CAN messages from the DBC file.
*   `fingerprints.py`: ECU firmware version database for identifying car models.
*   `interface.py`: High-level class for car interaction.
*   `radar_interface.py`: Handles radar data parsing.
*   `values.py`: Enumerates supported cars.

### Reverse Engineer CAN messages

Start by recording a drive with LKAS and ACC enabled, along with steering wheel movements, and load the route in [cabana](https://github.com/commaai/openpilot/tree/master/tools/cabana).

### Tuning

#### Longitudinal Control

Evaluate and tune your car's longitudinal control using the [longitudinal maneuvers](https://github.com/commaai/openpilot/tree/master/tools/longitudinal_maneuvers) report.

## Contributing

Contribute to opendbc development via GitHub and [Discord](https://discord.comma.ai). Engage in the `#dev-opendbc-cars` channel.

### Roadmap

*   **Short Term:**
    *   `pip install opendbc`
    *   100% type and line coverage
    *   Improved car port usability, tools, tests, and docs.
    *   Better state exposure for supported cars: https://github.com/commaai/opendbc/issues/1144
*   **Longer Term:**
    *   Expand support to all cars with LKAS and ACC interfaces.
    *   Automated lateral and longitudinal control/tuning evaluation.
    *   Auto-tuning for [lateral](https://blog.comma.ai/090release/#torqued-an-auto-tuner-for-lateral-control) and longitudinal control.
    *   [Automatic Emergency Braking](https://en.wikipedia.org/wiki/Automated_emergency_braking_system)

Contributions are welcome, focusing on these areas.

## Safety Model

The opendbc safety firmware, used with [openpilot](https://github.com/commaai/openpilot) and [panda](https://github.com/commaai/panda), defaults to `SAFETY_SILENT` mode, preventing bus activity. Safety modes enable message sending, with some modes (like `SAFETY_ALLOUTPUT`) disabled in release firmwares; custom builds are needed. These modes can optionally use `controls_allowed` for message control based on board state.

## Code Rigor

The safety firmware, critical for [openpilot](https://github.com/commaai/openpilot) and [panda](https://github.com/commaai/panda) operation, is rigorously tested within the `safety` folder.

*   **CI Regression Tests:**
    *   Static code analysis via [cppcheck](https://github.com/danmar/cppcheck/).
    *   [cppcheck](https://github.com/danmar/cppcheck/) checks for [MISRA C:2012](https://misra.org.uk/) violations. See [current coverage](opendbc/safety/tests/misra/coverage_table).
    *   Strict compiler flags: `-Wall -Wextra -Wstrict-prototypes -Werror`.
    *   [Safety logic](opendbc/safety) tested through [unit tests](opendbc/safety/tests).
*   **Testing of Tests:**
    *   [Mutation tests](opendbc/safety/tests/misra/test_mutation.py) on MISRA coverage.
    *   100% line coverage for safety unit tests.

In addition, the car interface library is tested with [ruff linter](https://github.com/astral-sh/ruff) and [mypy](https://mypy-lang.org/).

### Bounties

Get rewarded for your contribution! Bounties are available for:

*   $2000 - [Any car brand / platform port](https://github.com/orgs/commaai/projects/26/views/1?pane=issue&itemId=47913774)
*   $250 - [Any car model port](https://github.com/orgs/commaai/projects/26/views/1?pane=issue&itemId=47913790)
*   $300 - [Reverse Engineering a new Actuation Message](https://github.com/orgs/commaai/projects/26/views/1?pane=issue&itemId=73445563)

Higher value bounties also offered for popular cars, at [comma.ai/bounties](comma.ai/bounties).

## FAQ

*   **How do I use this?** The [comma 3X](https://comma.ai/shop/comma-3x) is the recommended hardware.
*   **Which cars are supported?** See the [supported cars list](docs/CARS.md).
*   **Can I add support for my car?** Yes, via community contribution, following the guide [here](https://github.com/commaai/opendbc/blob/docs/README.md#how-to-port-a-car).
*   **Which cars can be supported?** Any car with LKAS and ACC, more details [here](https://github.com/commaai/openpilot/blob/master/docs/CARS.md#dont-see-your-car-here).
*   **How does this work?** Hardware replaces built-in lane keep and adaptive cruise features. See this explanation: [How Do We Control The Car?](https://www.youtube.com/watch?v=nNU6ipme878&pp=ygUoY29tbWEgY29uIDIwMjEgaG93IGRvIHdlIGNvbnRyb2wgdGhlIGNhcg%3D%3D)
*   **Is there a timeline or roadmap for adding car support?**  Car support is largely community-driven, with comma doing validation.

### Terms

*   **port**: Car integration and support.
*   **lateral control**: Steering control.
*   **longitudinal control**: Gas/brakes control.
*   **fingerprinting**: Car identification process.
*   **[LKAS](https://en.wikipedia.org/wiki/Lane_departure_warning_system)**: Lane keeping assist.
*   **[ACC](https://en.wikipedia.org/wiki/Adaptive_cruise_control)**: Adaptive cruise control.
*   **[harness](https://comma.ai/shop/car-harness)**: Car-specific hardware.
*   **[panda](https://github.com/commaai/panda)**: CAN bus access hardware.
*   **[ECU](https://en.wikipedia.org/wiki/Electronic_control_unit)**: Car control modules.
*   **[CAN bus](https://en.wikipedia.org/wiki/CAN_bus)**: Car communication bus.
*   **[cabana](https://github.com/commaai/openpilot/tree/master/tools/cabana#readme)**: CAN message reverse engineering tool.
*   **[DBC file](https://en.wikipedia.org/wiki/CAN_bus#DBC)**: CAN message definition files.
*   **[openpilot](https://github.com/commaai/openpilot)**: ADAS system utilizing opendbc.
*   **[comma](https://github.com/commaai)**: The company behind opendbc.
*   **[comma 3X](https://comma.ai/shop/comma-3x)**: Hardware for running openpilot.

### More Resources

*   [*How Do We Control The Car?*](https://www.youtube.com/watch?v=nNU6ipme878&pp=ygUoY29tbWEgY29uIDIwMjEgaG93IGRvIHdlIGNvbnRyb2wgdGhlIGNhcg%3D%3D) by [@robbederks](https://github.com/robbederks) from COMMA_CON 2021
*   [*How to Port a Car*](https://www.youtube.com/watch?v=XxPS5TpTUnI&t=142s&pp=ygUPamFzb24gY29tbWEgY29u) by [@jyoung8607](https://github.com/jyoung8607) from COMMA_CON 2023
*   [commaCarSegments](https://huggingface.co/datasets/commaai/commaCarSegments): A massive dataset of CAN data from 300 different car models
*   [cabana](https://github.com/commaai/openpilot/tree/master/tools/cabana#readme): Our tool for reverse engineering CAN messages
*   [can_print_changes.py](https://github.com/commaai/openpilot/blob/master/selfdrive/debug/can_print_changes.py): Diff the whole CAN bus across two drives, such as one without any LKAS and one with LKAS
*   [longitudinal maneuvers](https://github.com/commaai/openpilot/tree/master/tools/longitudinal_maneuvers): A tool for evaluating and tuning longitudinal control
*   [opendbc data](https://commaai.github.io/opendbc-data/): A repository of longitudinal maneuver evaluations

## Join the Team - [comma.ai/jobs](https://comma.ai/jobs)

comma is hiring engineers to work on opendbc and [openpilot](https://github.com/commaai/openpilot). We value and encourage contributions.