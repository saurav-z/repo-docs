<div align="center" style="text-align: center;">

<h1>opendbc: Your Python API for Automotive Control & Data</h1>
<p>
  <b>Unlock your car's potential with opendbc, a powerful Python API for interacting with your vehicle's systems.</b>
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

opendbc is a Python API that gives you low-level access to your car's internal systems. This allows you to control gas, brake, steering, and read real-time data like speed and steering angle.  Originally developed to support the ADAS interfaces for [openpilot](https://github.com/commaai/openpilot), opendbc is expanding to read and write a wide range of car data, creating a versatile vehicle management tool.

**[Explore the opendbc Repository](https://github.com/commaai/opendbc)**

## Key Features

*   **Control:** Manipulate steering, gas, and brakes.
*   **Data Acquisition:** Read vehicle speed, steering angle, and more.
*   **Broad Compatibility:** Designed to support a wide range of vehicles with LKAS and ACC.
*   **Extensible:**  Easily integrate with and build upon your own projects.
*   **Community-Driven:**  Benefit from the contributions of a vibrant community.

## Quick Start

```bash
git clone https://github.com/commaai/opendbc.git
cd opendbc

# Run the all-in-one script
./test.sh

# Install dependencies, build, test, and lint individually
pip3 install -e .[testing,docs]  # install dependencies
scons -j8                        # build with 8 cores
pytest .                         # run the tests
lefthook run lint                # run the linter
```

Example programs are available in the [`examples/`](examples/) directory to help you get started.  [`examples/joystick.py`](examples/joystick.py) allows joystick control of your car.

## Project Structure

*   [`opendbc/dbc/`](opendbc/dbc/) - Contains [DBC](https://en.wikipedia.org/wiki/CAN_bus#DBC) files (CAN bus database files)
*   [`opendbc/can/`](opendbc/can/) - A library for parsing and building CAN messages.
*   [`opendbc/car/`](opendbc/car/) - High-level Python library for interfacing with cars.
*   [`opendbc/safety/`](opendbc/safety/) - Functional safety code for supported vehicles.

## How to Port a Car

Extend opendbc's functionality by adding support for new car models.

### Connect to the Car
Connect to your car via a [comma 3X](https://comma.ai/shop/comma-3x) and a car harness.  Harnesses for many vehicles are available at [comma.ai/shop](comma.ai/shop).

### Port Structure
A car port resides in `opendbc/car/<brand>/`:
*   `carstate.py`: Parses CAN data.
*   `carcontroller.py`: Sends CAN messages.
*   `<brand>can.py`: Helps build CAN messages.
*   `fingerprints.py`: Identifies car models.
*   `interface.py`: High-level interface class.
*   `radar_interface.py`: Parses radar data.
*   `values.py`: Lists supported car models.

### Reverse Engineer CAN Messages
Record a drive using [cabana](https://github.com/commaai/openpilot/tree/master/tools/cabana) to analyze CAN data.

### Tuning

Tune longitudinal control using the [longitudinal maneuvers](https://github.com/commaai/openpilot/tree/master/tools/longitudinal_maneuvers) report.

## Contributing

Contribute on GitHub and [Discord](https://discord.comma.ai) in the `#dev-opendbc-cars` channel.

### Roadmap

*   [ ] `pip install opendbc`
*   [ ] 100% type coverage
*   [ ] 100% line coverage
*   [ ] Simplify car port creation.
*   [ ] Improve state exposure of supported cars.
*   [ ] Expand support to all cars with LKAS and ACC.
*   [ ] Automate lateral and longitudinal control evaluation.
*   [ ] Implement auto-tuning for lateral and longitudinal control.

## Safety Model

The opendbc safety firmware, used with [openpilot](https://github.com/commaai/openpilot) and [panda](https://github.com/commaai/panda), provides safety features.  By default the CAN buses are silent.  Safety modes, such as `SAFETY_ALLOUTPUT`, are disabled in release firmwares.  Use your own build if you wish to enable these.

## Code Rigor

The safety firmware is written with high standards to ensure safety.
It includes the following [CI regression tests](https://github.com/commaai/opendbc/actions):
*   Static code analysis with [cppcheck](https://github.com/danmar/cppcheck/).
*   [MISRA C:2012](https://misra.org.uk/) violations are checked with [cppcheck](https://github.com/danmar/cppcheck/).
*   Strict compiler flags (`-Wall -Wextra -Wstrict-prototypes -Werror`).
*   [Unit tests](opendbc/safety/tests) for each car.
*   [Mutation test](opendbc/safety/tests/misra/test_mutation.py) of MISRA coverage.
*   100% line coverage on unit tests.
*   The [ruff linter](https://github.com/astral-sh/ruff) and [mypy](https://mypy-lang.org/) are run.

### Bounties

Earn bounties for your contributions:
*   $2000 - [Any car brand / platform port](https://github.com/orgs/commaai/projects/26/views/1?pane=issue&itemId=47913774)
*   $250 - [Any car model port](https://github.com/orgs/commaai/projects/26/views/1?pane=issue&itemId=47913790)
*   $300 - [Reverse Engineering a new Actuation Message](https://github.com/orgs/commaai/projects/26/views/1?pane=issue&itemId=73445563)

Check [comma.ai/bounties](comma.ai/bounties) for higher-value bounties.

## FAQ

*   **How do I use this?** A [comma 3X](https://comma.ai/shop/comma-3x) is recommended.
*   **Which cars are supported?** See the [supported cars list](docs/CARS.md).
*   **Can I add support for my car?** Yes, using the guide [here](https://github.com/commaai/opendbc/blob/docs/README.md#how-to-port-a-car).
*   **Which cars can be supported?** Cars with LKAS and ACC. More info [here](https://github.com/commaai/openpilot/blob/master/docs/CARS.md#dont-see-your-car-here).
*   **How does this work?** It replaces your car's lane keep and adaptive cruise features. See [this talk](https://www.youtube.com/watch?v=FL8CxUSfipM).
*   **Is there a timeline or roadmap for adding car support?** Community contributions are prioritized.

### Terms

*   **port**: Integration and support of a specific car
*   **lateral control**: Steering control
*   **longitudinal control**: Gas/brakes control
*   **fingerprinting**: Automatic car identification
*   **[LKAS](https://en.wikipedia.org/wiki/Lane_departure_warning_system)**: Lane keeping assist
*   **[ACC](https://en.wikipedia.org/wiki/Adaptive_cruise_control)**: Adaptive cruise control
*   **[harness](https://comma.ai/shop/car-harness)**: Hardware to attach to the car
*   **[panda](https://github.com/commaai/panda)**: Hardware for CAN bus access
*   **[ECU](https://en.wikipedia.org/wiki/Electronic_control_unit)**: Car control modules
*   **[CAN bus](https://en.wikipedia.org/wiki/CAN_bus)**: Car communication bus
*   **[cabana](https://github.com/commaai/openpilot/tree/master/tools/cabana#readme)**: Reverse engineering tool
*   **[DBC file](https://en.wikipedia.org/wiki/CAN_bus#DBC)**: CAN message definitions
*   **[openpilot](https://github.com/commaai/openpilot)**: ADAS system using opendbc
*   **[comma](https://github.com/commaai)**: The company behind opendbc
*   **[comma 3X](https://comma.ai/shop/comma-3x)**: Recommended hardware

### More resources

*   [*How Do We Control The Car?*](https://www.youtube.com/watch?v=nNU6ipme878&pp=ygUoY29tbWEgY29uIDIwMjEgaG93IGRvIHdlIGNvbnRyb2wgdGhlIGNhcg%3D%3D) by [@robbederks](https://github.com/robbederks) from COMMA_CON 2021
*   [*How to Port a Car*](https://www.youtube.com/watch?v=XxPS5TpTUnI&t=142s&pp=ygUPamFzb24gY29tbWEgY29u) by [@jyoung8607](https://github.com/jyoung8607) from COMMA_CON 2023
*   [commaCarSegments](https://huggingface.co/datasets/commaai/commaCarSegments): CAN data from 300 cars.
*   [cabana](https://github.com/commaai/openpilot/tree/master/tools/cabana#readme): Reverse engineering tool.
*   [can_print_changes.py](https://github.com/commaai/openpilot/blob/master/selfdrive/debug/can_print_changes.py): Diff the CAN bus.
*   [longitudinal maneuvers](https://github.com/commaai/openpilot/tree/master/tools/longitudinal_maneuvers): Longitudinal control tuning tool.
*   [opendbc data](https://commaai.github.io/opendbc-data/): Longitudinal maneuver evaluations.

## Join the Team! - [comma.ai/jobs](https://comma.ai/jobs)

comma is hiring engineers to work on opendbc and [openpilot](https://github.com/commaai/openpilot).  Contribute and potentially be hired.