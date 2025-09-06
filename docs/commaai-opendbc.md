<div align="center" style="text-align: center;">

<h1>opendbc: Python API for Your Car's ADAS</h1>
<p>
  <b>Control and read your car's systems with ease!</b>
  <br>
  opendbc provides a powerful Python API for accessing and controlling your vehicle's advanced driver-assistance systems (ADAS), like steering, gas, and brakes.
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

*   **Access and Control:** Interact with your car's steering, gas, brakes, and more.
*   **Real-Time Data:** Read crucial data like speed and steering angle.
*   **Extensive Car Support:** Designed to support a wide range of vehicles with LKAS and ACC systems.
*   **Community-Driven:** Benefit from a vibrant community and contribute to expanding car support.
*   **ADAS Focus:** Primarily supports ADAS features, enhancing functionality for projects like [openpilot](https://github.com/commaai/openpilot).
*   **Safety-Focused:** Includes a rigorous safety model to ensure responsible and reliable operation.

opendbc is a Python API that empowers you to take control of your car's systems.

[View the original repo on GitHub](https://github.com/commaai/opendbc)

## Quick Start

```bash
git clone https://github.com/commaai/opendbc.git
cd opendbc

# Run the all-in-one for dependency installation, compiling, linting, and tests.
./test.sh

# Alternatively, run individual commands:
pip3 install -e .[testing,docs]  # install dependencies
scons -j8                        # build with 8 cores
pytest .                         # run the tests
lefthook run lint                # run the linter
```

[`examples/`](examples/) provides sample programs to get you started, including a joystick control example: [`examples/joystick.py`](examples/joystick.py).

## Project Structure

*   [`opendbc/dbc/`](opendbc/dbc/): Stores [DBC](https://en.wikipedia.org/wiki/CAN_bus#DBC) files.
*   [`opendbc/can/`](opendbc/can/): Contains a library for parsing and building CAN messages.
*   [`opendbc/car/`](opendbc/car/): Provides a high-level Python library for interacting with cars.
*   [`opendbc/safety/`](opendbc/safety/): Implements functional safety features for supported vehicles.

## How to Port a Car

This guide walks you through adding support for new cars or enhancing existing ones:

### Connect to the Car

1.  Requires a [comma 3X](https://comma.ai/shop/comma-3x) and a car harness.
2.  Connect to two CAN buses, splitting one for actuation messages.
3.  Find compatible harnesses at [comma.ai/shop](https://comma.ai/shop), or use a developer harness.

### Structure of a Port

A car port typically resides in `opendbc/car/<brand>/` and includes:

*   `carstate.py`: Parses CAN data.
*   `carcontroller.py`: Sends control messages.
*   `<brand>can.py`: Builds CAN messages.
*   `fingerprints.py`: Identifies car models.
*   `interface.py`: High-level car interface.
*   `radar_interface.py`: Parses radar data.
*   `values.py`: Lists supported cars.

### Reverse Engineer CAN messages
Start by recording a route with interesting events and use [cabana](https://github.com/commaai/openpilot/tree/master/tools/cabana) to analyze the data.

### Tuning

#### Longitudinal

Use the [longitudinal maneuvers](https://github.com/commaai/openpilot/tree/master/tools/longitudinal_maneuvers) report to tune longitudinal control.

## Contributing

Contributions are welcome! Join the community on GitHub and [Discord](https://discord.comma.ai) in the `#dev-opendbc-cars` channel.

### Roadmap

*   **Short Term:**
    *   `pip install opendbc`
    *   100% type and line coverage.
    *   Improve car port ease.
    *   Better state exposition.
*   **Longer Term:**
    *   Expand support to all LKAS + ACC cars.
    *   Automated lateral/longitudinal control and tuning evaluation.
    *   Automated tuning for lateral and longitudinal control.

## Safety Model

The [panda](https://comma.ai/shop/panda) with opendbc safety firmware starts in `SAFETY_SILENT` mode, disabling CAN bus transmissions.  Safety modes (some disabled in release firmwares) must be selected to send messages. They may support `controls_allowed` for conditional message transmission.

## Code Rigor

The `safety` folder uses high standards for code quality, including:

*   Static code analysis (cppcheck).
*   MISRA C:2012 compliance.
*   Strict compiler flags (-Wall, -Wextra, etc.).
*   Unit tests with coverage and mutation tests.
*   Ruff linter and mypy for the car interface library.

## Bounties

Earn bounties for car porting contributions:

*   $2000 - [Any car brand / platform port](https://github.com/orgs/commaai/projects/26/views/1?pane=issue&itemId=47913774)
*   $250 - [Any car model port](https://github.com/orgs/commaai/projects/26/views/1?pane=issue&itemId=47913790)
*   $300 - [Reverse Engineering a new Actuation Message](https://github.com/orgs/commaai/projects/26/views/1?pane=issue&itemId=73445563)

Higher value bounties are available for popular cars at [comma.ai/bounties](comma.ai/bounties).

## FAQ

*   **How do I use this?** Requires a [comma 3X](https://comma.ai/shop/comma-3x).
*   **Which cars are supported?** See the [supported cars list](docs/CARS.md).
*   **Can I add support for my car?** Yes, the community drives car support; follow the guide [here](https://github.com/commaai/opendbc/blob/docs/README.md#how-to-port-a-car).
*   **Which cars can be supported?** Any car with LKAS and ACC. More info [here](https://github.com/commaai/openpilot/blob/master/docs/CARS.md#dont-see-your-car-here).
*   **How does this work?** Hardware replaces built-in lane keep and adaptive cruise features. See [this talk](https://www.youtube.com/watch?v=FL8CxUSfipM).
*   **Timeline/Roadmap for car support?** Community-driven, with comma validating for safety and quality.

### Terms

*   **port:** Car integration and support
*   **lateral control:** Steering control
*   **longitudinal control:** Gas/brakes control
*   **fingerprinting:** Automatic car identification
*   **[LKAS](https://en.wikipedia.org/wiki/Lane_departure_warning_system)**: Lane Keeping Assist
*   **[ACC](https://en.wikipedia.org/wiki/Adaptive_cruise_control)**: Adaptive Cruise Control
*   **[harness](https://comma.ai/shop/car-harness)**: Car-specific hardware attachment
*   **[panda](https://github.com/commaai/panda)**: CAN bus hardware
*   **[ECU](https://en.wikipedia.org/wiki/Electronic_control_unit)**: Car control modules
*   **[CAN bus](https://en.wikipedia.org/wiki/CAN_bus)**: ECU connection bus
*   **[cabana](https://github.com/commaai/openpilot/tree/master/tools/cabana#readme)**: Reverse engineering tool
*   **[DBC file](https://en.wikipedia.org/wiki/CAN_bus#DBC)**: CAN bus message definitions
*   **[openpilot](https://github.com/commaai/openpilot)**: ADAS system using opendbc
*   **[comma](https://github.com/commaai)**: The company behind opendbc
*   **[comma 3X](https://comma.ai/shop/comma-3x)**: Hardware for running openpilot

### More Resources

*   [*How Do We Control The Car?*](https://www.youtube.com/watch?v=nNU6ipme878&pp=ygUoY29tbWEgY29uIDIwMjEgaG93IGRvIHdlIGNvbnRyb2wgdGhlIGNhcg%3D%3D) by [@robbederks](https://github.com/robbederks)
*   [*How to Port a Car*](https://www.youtube.com/watch?v=XxPS5TpTUnI&t=142s&pp=ygUPamFzb24gY29tbWEgY29u) by [@jyoung8607](https://github.com/jyoung8607)
*   [commaCarSegments](https://huggingface.co/datasets/commaai/commaCarSegments): CAN data dataset
*   [cabana](https://github.com/commaai/openpilot/tree/master/tools/cabana#readme): Reverse engineering tool
*   [can_print_changes.py](https://github.com/commaai/openpilot/blob/master/selfdrive/debug/can_print_changes.py): CAN bus diff tool
*   [longitudinal maneuvers](https://github.com/commaai/openpilot/tree/master/tools/longitudinal_maneuvers): Longitudinal control tuning tool
*   [opendbc data](https://commaai.github.io/opendbc-data/): Longitudinal maneuver evaluations

## Come work with us -- [comma.ai/jobs](https://comma.ai/jobs)

Join comma! We're hiring engineers to work on opendbc and [openpilot](https://github.com/commaai/openpilot).