<div align="center">
  <h1>opendbc</h1>
  <p>
    <b>Unlock your car's potential with opendbc, the Python API for advanced vehicle control and data access.</b>
  </p>

  <p>
    <a href="https://docs.comma.ai">Docs</a>
    <span> · </span>
    <a href="https://github.com/commaai/openpilot/blob/master/docs/CONTRIBUTING.md">Contribute</a>
    <span> · </span>
    <a href="https://discord.comma.ai">Discord</a>
  </p>

  <p>
    <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT">
    <img src="https://img.shields.io/twitter/follow/comma_ai" alt="X Follow">
    <img src="https://img.shields.io/discord/469524606043160576" alt="Discord">
  </p>
</div>

---

opendbc empowers you to take control of your car's steering, gas, brakes, and more. This open-source project provides a Python API for interacting with your vehicle's Controller Area Network (CAN) bus, allowing you to read vehicle data and even send commands to control various functions. Built to support advanced driver-assistance systems (ADAS) like [openpilot](https://github.com/commaai/openpilot), opendbc aims to support a wide range of vehicles for comprehensive vehicle management. 

For more information on how to install and contribute, please visit the original repository at: [https://github.com/commaai/opendbc](https://github.com/commaai/opendbc).

**Key Features:**

*   🚗 **Control & Data:** Access and control steering, gas, brakes, and other vehicle functions.
*   🌐 **Open Source:** Leverage a community-driven project with an open-source MIT license.
*   ⚙️ **Vehicle Support:** Designed to support a wide array of vehicles with LKAS and ACC features.
*   🐍 **Python API:** Provides a user-friendly Python API for easy integration and development.
*   🛠️ **Community Driven:** Benefit from community contributions and a collaborative development environment.

## Getting Started

To quickly get started with opendbc, follow these steps:

```bash
git clone https://github.com/commaai/opendbc.git
cd opendbc

# Install dependencies, build, and run tests (recommended)
./test.sh

# Individual commands for more control
pip3 install -e .[testing,docs]  # Install dependencies
scons -j8                        # Build with 8 cores
pytest .                         # Run tests
lefthook run lint                # Run the linter
```

Explore the example programs in the [`examples/`](examples/) directory, such as [`examples/joystick.py`](examples/joystick.py), to interact with your car's features.

## Project Structure

*   [`opendbc/dbc/`](opendbc/dbc/): Stores [DBC](https://en.wikipedia.org/wiki/CAN_bus#DBC) files.
*   [`opendbc/can/`](opendbc/can/): Includes a library for parsing and building CAN messages.
*   [`opendbc/car/`](opendbc/car/): Houses a high-level library for interacting with cars using Python.
*   [`opendbc/safety/`](opendbc/safety/): Implements functional safety for all cars supported in `opendbc/car/`.

## How to Port a Car

Contribute to the project by adding support for new vehicles.

*   Connect to the Car: Use a comma 3X and a car harness to connect to the car.
*   Reverse Engineer CAN messages: Use tools like [cabana](https://github.com/commaai/openpilot/tree/master/tools/cabana) to analyze CAN data.

### Structure of a port

A car port lives in `opendbc/car/<brand>/`:
*   `carstate.py`: parses relevant CAN data.
*   `carcontroller.py`: Outputs CAN messages to control the car.
*   `<brand>can.py`: Python helpers to build CAN messages.
*   `fingerprints.py`: ECU firmware identification.
*   `interface.py`: High-level class for the car interface.
*   `radar_interface.py`: Parses out the radar.
*   `values.py`: Enumerates supported cars.

## Tuning

Use the [longitudinal maneuvers](https://github.com/commaai/openpilot/tree/master/tools/longitudinal_maneuvers) report to evaluate and tune longitudinal control.

## Contributing

Join the opendbc community! Coordinate with developers on GitHub and [Discord](https://discord.comma.ai) in the `#dev-opendbc-cars` channel and `Vehicle Specific` section.

### Roadmap

**Short Term:**

*   [ ] `pip install opendbc`
*   [ ] 100% type coverage
*   [ ] 100% line coverage
*   [ ] Simplify car ports: refactors, tools, tests, and docs
*   [ ] Improve state representation for all supported cars: https://github.com/commaai/opendbc/issues/1144

**Longer Term:**

*   [ ] Expand support for all LKAS + ACC equipped cars.
*   [ ] Implement automated lateral and longitudinal control/tuning evaluation.
*   [ ] Integrate auto-tuning for lateral and longitudinal control.
*   [ ] Implement Automatic Emergency Braking.

## Safety Model

The opendbc safety firmware is written for its use in conjunction with [openpilot](https://github.com/commaai/openpilot) and [panda](https://github.com/commaai/panda). The safety firmware, through its safety model, provides and enforces the
[openpilot safety](https://github.com/commaai/openpilot/blob/master/docs/SAFETY.md). Due to its critical function, it's important that the application code rigor within the `safety` folder is held to high standards.

The following tests are in place:
*   Generic static code analysis via [cppcheck](https://github.com/danmar/cppcheck/).
*   [MISRA C:2012](https://misra.org.uk/) violations are checked with a [cppcheck](https://github.com/danmar/cppcheck/) addon. See [current coverage](opendbc/safety/tests/misra/coverage_table).
*   Strict compiler options, using `-Wall -Wextra -Wstrict-prototypes -Werror` flags.
*   [Safety logic](opendbc/safety) tested and verified by [unit tests](opendbc/safety/tests) for each car variant.

The tests are tested by:
*   a [mutation test](opendbc/safety/tests/misra/test_mutation.py) on the MISRA coverage
*   100% line coverage enforced on the safety unit tests

Additional checks with [ruff linter](https://github.com/astral-sh/ruff) and [mypy](https://mypy-lang.org/) are used on the car interface library.

### Bounties

Contribute to opendbc and earn bounties:

*   $2000 - [Any car brand / platform port](https://github.com/orgs/commaai/projects/26/views/1?pane=issue&itemId=47913774)
*   $250 - [Any car model port](https://github.com/orgs/commaai/projects/26/views/1?pane=issue&itemId=47913790)
*   $300 - [Reverse Engineering a new Actuation Message](https://github.com/orgs/commaai/projects/26/views/1?pane=issue&itemId=73445563)

Higher-value bounties are offered for more popular cars. Check [comma.ai/bounties](comma.ai/bounties) for details.

## FAQ

*   **How do I use this?** A [comma 3X](https://comma.ai/shop/comma-3x) is the recommended hardware for running and developing opendbc and openpilot.
*   **Which cars are supported?** See the [supported cars list](docs/CARS.md).
*   **Can I add support for my car?** Yes, community contributions are welcome. Follow the [car port guide](https://github.com/commaai/opendbc/blob/docs/README.md#how-to-port-a-car).
*   **Which cars can be supported?** Any car with LKAS and ACC. Find more details [here](https://github.com/commaai/openpilot/blob/master/docs/CARS.md#dont-see-your-car-here).
*   **How does this work?** This system is designed to replace built-in lane keep and adaptive cruise features. See [this talk](https://www.youtube.com/watch?v=FL8CxUSfipM) for an in-depth explanation.
*   **Is there a timeline or roadmap for adding car support?** No, most car support is community-driven, with comma performing safety and quality checks.

### Terms

*   **port**: Car integration.
*   **lateral control**: Steering control.
*   **longitudinal control**: Gas/brakes control.
*   **fingerprinting**: Automatic car identification.
*   **[LKAS](https://en.wikipedia.org/wiki/Lane_departure_warning_system)**: Lane keeping assist.
*   **[ACC](https://en.wikipedia.org/wiki/Adaptive_cruise_control)**: Adaptive cruise control.
*   **[harness](https://comma.ai/shop/car-harness)**: Car-specific hardware for ADAS message interception.
*   **[panda](https://github.com/commaai/panda)**: Hardware for CAN bus access.
*   **[ECU](https://en.wikipedia.org/wiki/Electronic_control_unit)**: Car control modules.
*   **[CAN bus](https://en.wikipedia.org/wiki/CAN_bus)**: Network connecting ECUs.
*   **[cabana](https://github.com/commaai/openpilot/tree/master/tools/cabana#readme)**: Tool for CAN message analysis.
*   **[DBC file](https://en.wikipedia.org/wiki/CAN_bus#DBC)**: CAN bus message definitions.
*   **[openpilot](https://github.com/commaai/openpilot)**: ADAS system for opendbc-supported cars.
*   **[comma](https://github.com/commaai)**: The company behind opendbc.
*   **[comma 3X](https://comma.ai/shop/comma-3x)**: Recommended hardware for running openpilot.

### More Resources

*   [*How Do We Control The Car?*](https://www.youtube.com/watch?v=nNU6ipme878&pp=ygUoY29tbWEgY29uIDIwMjEgaG93IGRvIHdlIGNvbnRyb2wgdGhlIGNhcg%3D%3D) by [@robbederks](https://github.com/robbederks) from COMMA_CON 2021
*   [*How to Port a Car*](https://www.youtube.com/watch?v=XxPS5TpTUnI&t=142s&pp=ygUPamFzb24gY29tbWEgY29u) by [@jyoung8607](https://github.com/jyoung8607) from COMMA_CON 2023
*   [commaCarSegments](https://huggingface.co/datasets/commaai/commaCarSegments): a massive dataset of CAN data from 300 different car models
*   [cabana](https://github.com/commaai/openpilot/tree/master/tools/cabana#readme): our tool for reverse engineering CAN messages
*   [can_print_changes.py](https://github.com/commaai/openpilot/blob/master/selfdrive/debug/can_print_changes.py): diff the whole CAN bus across two drives, such as one without any LKAS and one with LKAS
*   [longitudinal maneuvers](https://github.com/commaai/openpilot/tree/master/tools/longitudinal_maneuvers): a tool for evaluating and tuning longitudinal control
*   [opendbc data](https://commaai.github.io/opendbc-data/): a repository of longitudinal maneuver evaluations

## Join the Team - [comma.ai/jobs](https://comma.ai/jobs)

comma is actively hiring engineers to work on opendbc and [openpilot](https://github.com/commaai/openpilot), and we welcome contributions!