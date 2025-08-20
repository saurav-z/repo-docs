<div align="center" style="text-align: center;">

<h1>opendbc: Your Python API for Vehicle Control</h1>
<p>
  <b>Unlock advanced control and data access for your car with opendbc!</b>
  <br>
  Take control of steering, acceleration, braking, and more with this powerful Python API.
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

opendbc provides a powerful Python API for interacting with your vehicle's Controller Area Network (CAN) bus, enabling you to control and monitor various aspects of your car. This open-source project, primarily used for [openpilot](https://github.com/commaai/openpilot) development, allows you to manipulate steering, acceleration, braking, and access real-time data from your vehicle.  **Explore the power of open vehicle control – [get started with opendbc](https://github.com/commaai/opendbc)!**

## Key Features

*   **Comprehensive Vehicle Control:** Easily control steering, gas, brakes, and more.
*   **Real-time Data Access:** Read speed, steering angle, and other critical vehicle data.
*   **Extensive Car Support:**  Supports a growing list of vehicles with LKAS and ACC systems.
*   **Open-Source & Community-Driven:** Benefit from a collaborative environment for development and innovation.
*   **Well-Documented:** Comprehensive documentation, community support and example code make it easy to get started and contribute.
*   **Safety Focused:**  Rigorous code testing and a dedicated safety model ensure reliable and secure operation.

## Quick Start

Get up and running quickly with these simple commands:

```bash
git clone https://github.com/commaai/opendbc.git
cd opendbc

# Install dependencies, build, lint, and run tests
./test.sh
```

For individual steps:

```bash
pip3 install -e .[testing,docs]  # install dependencies
scons -j8                        # build with 8 cores
pytest .                         # run the tests
lefthook run lint                # run the linter
```

Explore the `examples/` directory for practical demonstrations, including `examples/joystick.py` for joystick-controlled vehicle operation.

## Project Structure

*   `opendbc/dbc/`:  Contains [DBC](https://en.wikipedia.org/wiki/CAN_bus#DBC) files, which define CAN message structures.
*   `opendbc/can/`:  A library for parsing and building CAN messages from DBC files.
*   `opendbc/car/`: A high-level Python library for interacting with cars.
*   `opendbc/safety/`:  Provides functional safety mechanisms for supported vehicles.

## How to Port a Car

Learn how to add support for your car or improve existing support.

1.  **Connect to the Car:** Utilize a comma 3X and a compatible car harness. Developer harnesses are available at comma.ai/shop.
2.  **Structure of a Port:**  Each car port resides in `opendbc/car/<brand>/` and includes files for car state parsing, control output, CAN message helpers, fingerprinting, interface management, and radar parsing.
3.  **Reverse Engineer CAN messages:** Use tools like [cabana](https://github.com/commaai/openpilot/tree/master/tools/cabana) to analyze and understand CAN communication.
4.  **Tuning:** Use the [longitudinal maneuvers](https://github.com/commaai/openpilot/tree/master/tools/longitudinal_maneuvers) report to evaluate and tune longitudinal control.

## Contributing

Contribute to opendbc on GitHub and Discord.  Join the `#dev-opendbc-cars` channel for discussions and assistance.

### Roadmap

Short term:

*   Implement `pip install opendbc`.
*   Achieve 100% type and line coverage.
*   Improve car port development with refactoring, tools, tests, and documentation.
*   Enhance car state visibility (see issue: [#1144](https://github.com/commaai/opendbc/issues/1144)).

Longer term:

*   Expand support to all cars with LKAS and ACC interfaces.
*   Automate lateral and longitudinal control and tuning evaluation.
*   Develop auto-tuning capabilities for lateral and longitudinal control.
*   Implement Automatic Emergency Braking.

Contributions are welcomed for all these areas!

## Safety Model

The opendbc safety firmware, in conjunction with [openpilot](https://github.com/commaai/openpilot) and [panda](https://comma.ai/shop/panda), employs a safety model. This model, by default in `SAFETY_SILENT` mode, restricts CAN bus activity until a safety mode is selected. Certain modes are disabled in release firmware; custom builds are required for their use.

## Code Rigor

The safety firmware within opendbc prioritizes rigorous application code standards:

*   Static code analysis via [cppcheck](https://github.com/danmar/cppcheck/) with [MISRA C:2012](https://misra.org.uk/) compliance (see [coverage details](opendbc/safety/tests/misra/coverage_table)).
*   Strict compiler flags: `-Wall -Wextra -Wstrict-prototypes -Werror`.
*   Testing: [Unit tests](opendbc/safety/tests) with 100% line coverage.
*   Mutation testing for MISRA coverage.
*   Use of [ruff linter](https://github.com/astral-sh/ruff) and [mypy](https://mypy-lang.org/) on car interface libraries.

### Bounties

Earn bounties for contributions:

*   $2000 -  [Any car brand / platform port](https://github.com/orgs/commaai/projects/26/views/1?pane=issue&itemId=47913774)
*   $250 - [Any car model port](https://github.com/orgs/commaai/projects/26/views/1?pane=issue&itemId=47913790)
*   $300 - [Reverse Engineering a new Actuation Message](https://github.com/orgs/commaai/projects/26/views/1?pane=issue&itemId=73445563)

Higher value bounties are available for popular car models at [comma.ai/bounties](comma.ai/bounties).

## FAQ

***How do I use this?*** Use a [comma 3X](https://comma.ai/shop/comma-3x), designed for opendbc and openpilot development.

***Which cars are supported?*** See the [supported cars list](docs/CARS.md).

***Can I add support for my car?*** Yes, community contributions are welcome!  Follow the guide [here](https://github.com/commaai/opendbc/blob/docs/README.md#how-to-port-a-car).

***Which cars can be supported?*** Any car with LKAS and ACC.  More info [here](https://github.com/commaai/openpilot/blob/master/docs/CARS.md#dont-see-your-car-here).

***How does this work?***  opendbc interfaces with your car's ADAS features. Watch this [talk](https://www.youtube.com/watch?v=FL8CxUSfipM).

***Is there a timeline or roadmap for adding car support?*** Community-driven, with comma validating contributions.

### Terms

*   **port**: Car integration and support.
*   **lateral control**: Steering control.
*   **longitudinal control**: Gas/brakes control.
*   **fingerprinting**: Automatic car identification.
*   **[LKAS](https://en.wikipedia.org/wiki/Lane_departure_warning_system)**: Lane keeping assist.
*   **[ACC](https://en.wikipedia.org/wiki/Adaptive_cruise_control)**: Adaptive cruise control.
*   **[harness](https://comma.ai/shop/car-harness)**: Car-specific ADAS message interception hardware.
*   **[panda](https://github.com/commaai/panda)**: Hardware for CAN bus access.
*   **[ECU](https://en.wikipedia.org/wiki/Electronic_control_unit)**: In-car computers/control modules.
*   **[CAN bus](https://en.wikipedia.org/wiki/CAN_bus)**: Inter-ECU communication bus.
*   **[cabana](https://github.com/commaai/openpilot/tree/master/tools/cabana#readme)**: CAN message reverse engineering tool.
*   **[DBC file](https://en.wikipedia.org/wiki/CAN_bus#DBC)**: CAN message definitions.
*   **[openpilot](https://github.com/commaai/openpilot)**: ADAS system.
*   **[comma](https://github.com/commaai)**: The company behind opendbc.
*   **[comma 3X](https://comma.ai/shop/comma-3x)**:  Hardware to run openpilot.

### More Resources

*   [*How Do We Control The Car?*](https://www.youtube.com/watch?v=nNU6ipme878&pp=ygUoY29tbWEgY29uIDIwMjEgaG93IGRvIHdlIGNvbnRyb2wgdGhlIGNhcg%3D%3D) by [@robbederks](https://github.com/robbederks) from COMMA_CON 2021
*   [*How to Port a Car*](https://www.youtube.com/watch?v=XxPS5TpTUnI&t=142s&pp=ygUPamFzb24gY29tbWEgY29u) by [@jyoung8607](https://github.com/jyoung8607) from COMMA_CON 2023
*   [commaCarSegments](https://huggingface.co/datasets/commaai/commaCarSegments): CAN data from 300 car models.
*   [cabana](https://github.com/commaai/openpilot/tree/master/tools/cabana#readme): Reverse engineering CAN messages.
*   [can_print_changes.py](https://github.com/commaai/openpilot/blob/master/selfdrive/debug/can_print_changes.py):  Compare CAN bus data across drives.
*   [longitudinal maneuvers](https://github.com/commaai/openpilot/tree/master/tools/longitudinal_maneuvers): Longitudinal control evaluation.
*   [opendbc data](https://commaai.github.io/opendbc-data/): Longitudinal maneuver evaluations.

## Join the Team – [comma.ai/jobs](https://comma.ai/jobs)

comma is hiring engineers to work on opendbc and [openpilot](https://github.com/commaai/openpilot).  We actively seek community contributors!