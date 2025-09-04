<div align="center" style="text-align: center;">

<h1>opendbc</h1>
<p>
  <b>Unlock your car's potential: opendbc is the open-source Python API that puts you in control.</b>
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

opendbc is a powerful Python API designed to give you unprecedented control over your vehicle's systems, offering a flexible and open-source platform for accessing and manipulating car data. With opendbc, you can read and control various aspects of your car, including steering, acceleration, braking, and more. 

**[Explore the opendbc project on GitHub](https://github.com/commaai/opendbc)**

**Key Features:**

*   **Control & Read:** Control gas, brake, steering, and read critical data like speed and steering angle.
*   **Broad Compatibility:** Designed to support a wide range of vehicles with LKAS and ACC systems.
*   **Open Source:** Benefit from a collaborative community and contribute to the project's development.
*   **ADAS Focus:** Primary focus on ADAS interfaces, allowing you to build the best vehicle management app ever.
*   **Community Driven:** Heavily reliant on the community to support various cars.

---

## Getting Started

opendbc provides everything you need to get started, from car communication to car control.

```bash
git clone https://github.com/commaai/opendbc.git
cd opendbc

# Installation, compilation, linting, and tests in one go!
./test.sh

# Install dependencies
pip3 install -e .[testing,docs]
# Build
scons -j8
# Run the tests
pytest .
# Run the linter
lefthook run lint
```

Example programs in the [`examples/`](examples/) directory provide useful samples and can read car state and control steering, gas, and brakes. Control a car with a joystick using [`examples/joystick.py`](examples/joystick.py).

### Project Structure

*   [`opendbc/dbc/`](opendbc/dbc/) : Contains [DBC](https://en.wikipedia.org/wiki/CAN_bus#DBC) files, crucial for understanding vehicle communication.
*   [`opendbc/can/`](opendbc/can/) : A library for parsing and building CAN messages from DBC files.
*   [`opendbc/car/`](opendbc/car/) : High-level library for interfacing with cars using Python.
*   [`opendbc/safety/`](opendbc/safety/) : Security features and functional safety for all supported cars.

## How to Port a Car

Extensive documentation for creating support for new vehicles can be found in this repository. A complete car port will have all the functionalities of: lateral control, longitudinal control, radar parsing (if equipped), fuzzy fingerprinting, and more.

### Connect to the Car

1.  **Connect:** Get connected to the car using a comma 3X and a car harness.
2.  **Harness:** The car harness connects to two CAN buses to send custom actuation messages.
3.  **Developer Harness:** If your car's harness doesn't exist, crimp on a developer harness from comma.ai/shop.

### Structure of a Port

The basic structure of a car port is typically found in `opendbc/car/<brand>/`:

*   `carstate.py`: Parses information from the CAN stream using the car's DBC file.
*   `carcontroller.py`: Outputs CAN messages to control the car.
*   `<brand>can.py`: Python helpers built on DBC files for CAN message construction.
*   `fingerprints.py`: Database of ECU firmware versions for identifying car models.
*   `interface.py`: High-level class for interfacing with the car.
*   `radar_interface.py`: Parses radar data.
*   `values.py`: Enumerates the brand's supported cars.

### Reverse Engineer CAN messages

Use [cabana](https://github.com/commaai/openpilot/tree/master/tools/cabana) to reverse engineer CAN messages. Record a route with interesting events, such as engaging LKAS and ACC, and then analyze the route data.

### Tuning

Use the [longitudinal maneuvers](https://github.com/commaai/openpilot/tree/master/tools/longitudinal_maneuvers) report to assess your car's longitudinal control and fine-tune it.

## Contributing

opendbc thrives on community contributions! All development is coordinated on GitHub and [Discord](https://discord.comma.ai). Engage in the `#dev-opendbc-cars` channel and `Vehicle Specific` section.

### Roadmap

*   **Short Term:**
    *   Implement `pip install opendbc`.
    *   Ensure 100% type and line coverage.
    *   Streamline car port creation with refactors, tools, tests, and documentation.
    *   Improve state access for supported cars.
*   **Longer Term:**
    *   Extend support to all cars with LKAS and ACC interfaces.
    *   Automate lateral and longitudinal control/tuning evaluation.
    *   Enable auto-tuning for lateral and longitudinal control.
    *   Incorporate Automated Emergency Braking.

## Safety Model

When used with the [panda](https://comma.ai/shop/panda) and [opendbc safety firmware](opendbc/safety), `opendbc` defaults to `SAFETY_SILENT` mode, which prevents CAN bus communication. Select a safety mode to send messages; some modes are disabled in release firmwares, so you must compile and flash your own build.

## Code Rigor

The safety firmware, used in conjunction with [openpilot](https://github.com/commaai/openpilot) and [panda](https://github.com/commaai/panda), is developed with high code rigor standards. This safety model provides and enforces [openpilot safety](https://github.com/commaai/openpilot/blob/master/docs/SAFETY.md).

Here are the [CI regression tests](https://github.com/commaai/opendbc/actions):

*   A generic static code analysis is performed by [cppcheck](https://github.com/danmar/cppcheck/).
*   [cppcheck](https://github.com/danmar/cppcheck/) has a specific addon to check for [MISRA C:2012](https://misra.org.uk/) violations.
*   Compiler options: `-Wall -Wextra -Wstrict-prototypes -Werror`.
*   [safety logic](opendbc/safety) is tested and verified by [unit tests](opendbc/safety/tests).

These tests are evaluated by:
*   a [mutation test](opendbc/safety/tests/misra/test_mutation.py)
*   100% line coverage for the safety unit tests

In addition, we run the [ruff linter](https://github.com/astral-sh/ruff) and [mypy](https://mypy-lang.org/) on the car interface library.

### Bounties

Earn bounties for contributing!

*   $2000 - [Any car brand / platform port](https://github.com/orgs/commaai/projects/26/views/1?pane=issue&itemId=47913774)
*   $250 - [Any car model port](https://github.com/orgs/commaai/projects/26/views/1?pane=issue&itemId=47913790)
*   $300 - [Reverse Engineering a new Actuation Message](https://github.com/orgs/commaai/projects/26/views/1?pane=issue&itemId=73445563)

See [comma.ai/bounties](comma.ai/bounties) for higher-value bounties on specific car models.

## FAQ

*   **How do I use this?**
    *   A [comma 3X](https://comma.ai/shop/comma-3x) is recommended for opendbc and openpilot development.
*   **Which cars are supported?**
    *   See the [supported cars list](docs/CARS.md).
*   **Can I add support for my car?**
    *   Yes! Follow the guide [here](https://github.com/commaai/opendbc/blob/docs/README.md#how-to-port-a-car).
*   **Which cars can be supported?**
    *   Any car with LKAS and ACC. More info [here](https://github.com/commaai/openpilot/blob/master/docs/CARS.md#dont-see-your-car-here).
*   **How does this work?**
    *   We designed hardware to replace your car's lane keep and adaptive cruise features. See [this talk](https://www.youtube.com/watch?v=FL8CxUSfipM).
*   **Is there a timeline or roadmap for adding car support?**
    *   Most car support comes from the community, with comma doing final safety and quality validation.

### Glossary

*   **port**: Car-specific integration and support.
*   **lateral control**: Steering control.
*   **longitudinal control**: Gas/brakes control.
*   **fingerprinting**: Automatic car identification.
*   **[LKAS](https://en.wikipedia.org/wiki/Lane_departure_warning_system)**: Lane Keeping Assist.
*   **[ACC](https://en.wikipedia.org/wiki/Adaptive_cruise_control)**: Adaptive Cruise Control.
*   **[harness](https://comma.ai/shop/car-harness)**: Car-specific hardware for ADAS message interception.
*   **[panda](https://github.com/commaai/panda)**: Hardware for CAN bus access.
*   **[ECU](https://en.wikipedia.org/wiki/Electronic_control_unit)**: Car's internal control modules.
*   **[CAN bus](https://en.wikipedia.org/wiki/CAN_bus)**: Car's communication network.
*   **[cabana](https://github.com/commaai/openpilot/tree/master/tools/cabana#readme)**: CAN message reverse engineering tool.
*   **[DBC file](https://en.wikipedia.org/wiki/CAN_bus#DBC)**: Defines CAN bus messages.
*   **[openpilot](https://github.com/commaai/openpilot)**: ADAS system supported by opendbc.
*   **[comma](https://github.com/commaai)**: The company behind opendbc.
*   **[comma 3X](https://comma.ai/shop/comma-3x)**: Hardware for running openpilot.

### More resources

*   [*How Do We Control The Car?*](https://www.youtube.com/watch?v=nNU6ipme878&pp=ygUoY29tbWEgY29uIDIwMjEgaG93IGRvIHdlIGNvbnRyb2wgdGhlIGNhcg%3D%3D) by [@robbederks](https://github.com/robbederks) from COMMA_CON 2021
*   [*How to Port a Car*](https://www.youtube.com/watch?v=XxPS5TpTUnI&t=142s&pp=ygUPamFzb24gY29tbWEgY29u) by [@jyoung8607](https://github.com/jyoung8607) from COMMA_CON 2023
*   [commaCarSegments](https://huggingface.co/datasets/commaai/commaCarSegments): a massive dataset of CAN data from 300 different car models
*   [cabana](https://github.com/commaai/openpilot/tree/master/tools/cabana#readme): our tool for reverse engineering CAN messages
*   [can_print_changes.py](https://github.com/commaai/openpilot/blob/master/selfdrive/debug/can_print_changes.py): diff the whole CAN bus across two drives, such as one without any LKAS and one with LKAS
*   [longitudinal maneuvers](https://github.com/commaai/openpilot/tree/master/tools/longitudinal_maneuvers): a tool for evaluating and tuning longitudinal control
*   [opendbc data](https://commaai.github.io/opendbc-data/): a repository of longitudinal maneuver evaluations

## Come work with us -- [comma.ai/jobs](https://comma.ai/jobs)

Join the team! comma is hiring engineers to work on opendbc and [openpilot](https://github.com/commaai/openpilot).