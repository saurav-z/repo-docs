<div align="center">

<h1>opendbc</h1>
<p>
  <b>opendbc: Unleash your car's potential with a Python API for advanced vehicle control and data access.</b>
  <br>
  <a href="https://github.com/commaai/opendbc"> <img src="https://img.shields.io/github/stars/commaai/opendbc?style=social" alt="Stars"> </a>
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

opendbc is a powerful Python API, allowing you to control and monitor your car's systems, providing access to critical data like speed and steering angle. This project aims to make controlling steering, gas, and brakes on supported vehicles simple and extensible.

**Key Features:**

*   **Control & Read:** Read and write data from your vehicle, including gas, brake, and steering.
*   **Broad Support:** Works with many cars equipped with LKAS and ACC.
*   **Open Source:** Benefit from community contributions and extend the API's functionality.
*   **Extensive Documentation:** Comprehensive guides for using, contributing, and extending opendbc are available.

## Getting Started

Quickly set up and test opendbc with these commands:

```bash
git clone https://github.com/commaai/opendbc.git
cd opendbc

# All-in-one for dependency installation, compiling, linting, and tests:
./test.sh

# Individual commands:
pip3 install -e .[testing,docs]  # Install dependencies
scons -j8                        # Build with 8 cores
pytest .                         # Run the tests
lefthook run lint                # Run the linter
```

Explore the examples in the [`examples/`](examples/) directory to read car state and control steering, gas, and brakes. Use [`examples/joystick.py`](examples/joystick.py) to control a car with a joystick.

## Project Structure

*   [`opendbc/dbc/`](opendbc/dbc/) - Stores [DBC](https://en.wikipedia.org/wiki/CAN_bus#DBC) files, containing data descriptions for different car models.
*   [`opendbc/can/`](opendbc/can/) - Provides a library for parsing and constructing CAN messages from DBC files.
*   [`opendbc/car/`](opendbc/car/) -  Offers a high-level Python interface for interacting with vehicle systems.
*   [`opendbc/safety/`](opendbc/safety/) -  Implements the functional safety mechanisms for all supported car brands, ensuring safe operation.

## Contributing

opendbc thrives on community contributions.  Engage with the project on GitHub and [Discord](https://discord.comma.ai) in the `#dev-opendbc-cars` channel.

## How to Port a Car

Learn how to add support for your car, from basic steering control to full ADAS integration.

### Connect to the Car

Connect to the car using a comma 3X and a car harness. Car harnesses connect to CAN buses and split a bus to send actuation messages.

### Structure of a port

*   `carstate.py` - Parses data from the CAN stream.
*   `carcontroller.py` - Sends CAN messages to control the car.
*   `<brand>can.py` - Helper functions for building CAN messages.
*   `fingerprints.py` - Database of ECU firmware versions.
*   `interface.py` - High-level class for car interaction.
*   `radar_interface.py` - Parses radar data.
*   `values.py` - Defines supported car models.

### Reverse Engineer CAN messages

Use [cabana](https://github.com/commaai/openpilot/tree/master/tools/cabana) to analyze CAN data.

### Tuning

Use the [longitudinal maneuvers](https://github.com/commaai/openpilot/tree/master/tools/longitudinal_maneuvers) report to evaluate your car's longitudinal control.

## Roadmap

The project roadmap includes:

*   `pip install opendbc`
*   100% type and line coverage
*   Improve car port development
*   Expand car support
*   Automated control/tuning evaluation
*   Auto-tuning for lateral and longitudinal control
*   Automatic Emergency Braking

## Safety Model

The safety firmware enforces the [openpilot safety](https://github.com/commaai/openpilot/blob/master/docs/SAFETY.md) model.

## Code Rigor

The opendbc safety firmware utilizes rigorous testing and analysis, including static code analysis, MISRA C:2012 compliance checks, strict compiler flags, unit tests with 100% line coverage, and mutation testing.

## Bounties

Get rewarded for your contributions:

*   $2000 - [Any car brand / platform port](https://github.com/orgs/commaai/projects/26/views/1?pane=issue&itemId=47913774)
*   $250 - [Any car model port](https://github.com/orgs/commaai/projects/26/views/1?pane=issue&itemId=47913790)
*   $300 - [Reverse Engineering a new Actuation Message](https://github.com/orgs/commaai/projects/26/views/1?pane=issue&itemId=73445563)

See [comma.ai/bounties](comma.ai/bounties) for additional bounties.

## FAQ

*   **How do I use this?** A [comma 3X](https://comma.ai/shop/comma-3x) is recommended.
*   **Which cars are supported?**  See the [supported cars list](docs/CARS.md).
*   **Can I add support for my car?** Yes, see the car port guide [here](https://github.com/commaai/opendbc/blob/docs/README.md#how-to-port-a-car).
*   **Which cars can be supported?** Any car with LKAS and ACC. More info [here](https://github.com/commaai/openpilot/blob/master/docs/CARS.md#dont-see-your-car-here).
*   **How does this work?** See [this talk](https://www.youtube.com/watch?v=FL8CxUSfipM).
*   **Is there a timeline or roadmap for adding car support?** Community-driven, validated by comma.

### Glossary

*   **port:** Car integration and support
*   **lateral control:** Steering control
*   **longitudinal control:** Gas/Brakes control
*   **fingerprinting:** Car identification
*   **[LKAS](https://en.wikipedia.org/wiki/Lane_departure_warning_system):** Lane keeping assist
*   **[ACC](https://en.wikipedia.org/wiki/Adaptive_cruise_control):** Adaptive cruise control
*   **[harness](https://comma.ai/shop/car-harness):** Car-specific hardware
*   **[panda](https://github.com/commaai/panda):** Hardware to access CAN bus
*   **[ECU](https://en.wikipedia.org/wiki/Electronic_control_unit):** Car's computer
*   **[CAN bus](https://en.wikipedia.org/wiki/CAN_bus):** Car communication network
*   **[cabana](https://github.com/commaai/openpilot/tree/master/tools/cabana#readme):** Reverse engineering tool
*   **[DBC file](https://en.wikipedia.org/wiki/CAN_bus#DBC):** CAN bus message definitions
*   **[openpilot](https://github.com/commaai/openpilot):** ADAS system using opendbc
*   **[comma](https://github.com/commaai):** The company behind opendbc
*   **[comma 3X](https://comma.ai/shop/comma-3x):** Hardware to run openpilot

### More resources

*   [*How Do We Control The Car?*](https://www.youtube.com/watch?v=nNU6ipme878&pp=ygUoY29tbWEgY29uIDIwMjEgaG93IGRvIHdlIGNvbnRyb2wgdGhlIGNhcg%3D%3D) by [@robbederks](https://github.com/robbederks) from COMMA_CON 2021
*   [*How to Port a Car*](https://www.youtube.com/watch?v=XxPS5TpTUnI&t=142s&pp=ygUPamFzb24gY29tbWEgY29u) by [@jyoung8607](https://github.com/jyoung8607) from COMMA_CON 2023
*   [commaCarSegments](https://huggingface.co/datasets/commaai/commaCarSegments): a massive dataset of CAN data from 300 different car models
*   [cabana](https://github.com/commaai/openpilot/tree/master/tools/cabana#readme): our tool for reverse engineering CAN messages
*   [can_print_changes.py](https://github.com/commaai/openpilot/blob/master/selfdrive/debug/can_print_changes.py): diff the whole CAN bus across two drives, such as one without any LKAS and one with LKAS
*   [longitudinal maneuvers](https://github.com/commaai/openpilot/tree/master/tools/longitudinal_maneuvers): a tool for evaluating and tuning longitudinal control
*   [opendbc data](https://commaai.github.io/opendbc-data/): a repository of longitudinal maneuver evaluations

## Join the Team! -- [comma.ai/jobs](https://comma.ai/jobs)

comma is hiring! Contribute and work on opendbc and [openpilot](https://github.com/commaai/openpilot).