<div align="center" style="text-align: center;">

<h1>opendbc: Open-Source Python API for Your Car</h1>
<p>
  <b>Take control of your vehicle with opendbc, a powerful Python API for accessing and controlling your car's systems.</b>
  <br>
  Unlock the potential to manipulate gas, brakes, steering, and gather real-time data like speed and steering angle.
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

*   **Control:** Actuate steering, gas, and brakes.
*   **Data Acquisition:** Read real-time car data (speed, steering angle, and more).
*   **Open Source:** Contribute to a community-driven project.
*   **ADAS Focus:** Designed to support ADAS interfaces, primarily for openpilot.
*   **Extensible:** Read and write various car systems like EV charge status, and door locks.

Discover the power to interact with your vehicle's systems directly through opendbc, an open-source Python API.

---

opendbc empowers developers and automotive enthusiasts to access and control various aspects of modern vehicles.  Primarily used to support [openpilot](https://github.com/commaai/openpilot), the project also allows for reading and writing data from your car.

## Getting Started

Clone the repository, install dependencies, and run tests with these commands:

```bash
git clone https://github.com/commaai/opendbc.git
cd opendbc
./test.sh
```

Explore example programs in the [`examples/`](examples/) directory for practical usage. For example, control your car with a joystick using [`examples/joystick.py`](examples/joystick.py).

## Project Structure

*   [`opendbc/dbc/`](opendbc/dbc/) : Stores [DBC](https://en.wikipedia.org/wiki/CAN_bus#DBC) files for different car models.
*   [`opendbc/can/`](opendbc/can/) : Contains a library for parsing and building CAN messages using DBC files.
*   [`opendbc/car/`](opendbc/car/) : Provides a high-level Python library for interacting with cars.
*   [`opendbc/safety/`](opendbc/safety/) : Implements functional safety measures for supported vehicles.

## How to Port a Car

Contribute to the project by adding support for new car models.  This guide explains the process, from initial setup to advanced tuning, to support controlling steering, gas, and brakes.

*   **Connect to the Car:** Use a [comma 3X](https://comma.ai/shop/comma-3x) and car harness for connection.
*   **Structure of a Port:** Understand the structure of a car port within `opendbc/car/<brand>/`.
*   **Reverse Engineer CAN Messages:** Use tools like [cabana](https://github.com/commaai/openpilot/tree/master/tools/cabana) to analyze and understand CAN data.
*   **Tuning:** Utilize tools like the [longitudinal maneuvers](https://github.com/commaai/openpilot/tree/master/tools/longitudinal_maneuvers) report for tuning.

## Contributing

Join the opendbc community on GitHub and [Discord](https://discord.comma.ai) to collaborate.

### Roadmap

*   **Short Term:**
    *   `pip install opendbc`
    *   100% type and line coverage.
    *   Improve car port creation with refactors, tools, tests, and docs.
    *   Better state exposure for supported cars.
*   **Longer Term:**
    *   Expand support to include every car with LKAS + ACC interfaces.
    *   Automated lateral and longitudinal control and tuning evaluation.
    *   Auto-tuning for lateral and longitudinal control.
    *   Implement Automated Emergency Braking.

Contributions are welcome!

## Safety Model

The opendbc safety firmware is used with [openpilot](https://github.com/commaai/openpilot) and [panda](https://github.com/commaai/panda). The `SAFETY_SILENT` mode is used by default when a [panda](https://comma.ai/shop/panda) is powered up, forcing the CAN buses to be silent. You have to select a safety mode to send messages.

## Code Rigor

The code rigor within the `safety` folder is held to high standards.

*   Static code analysis is performed by [cppcheck](https://github.com/danmar/cppcheck/) and [MISRA C:2012](https://misra.org.uk/) violations, including [current coverage](opendbc/safety/tests/misra/coverage_table).
*   Strict compiler options are used, with `-Wall -Wextra -Wstrict-prototypes -Werror`.
*   The safety logic is tested by [unit tests](opendbc/safety/tests) for each supported car variant.
*   Tests include a [mutation test](opendbc/safety/tests/misra/test_mutation.py) on MISRA coverage and 100% line coverage on the safety unit tests.
*   The [ruff linter](https://github.com/astral-sh/ruff) and [mypy](https://mypy-lang.org/) are run on the car interface library.

### Bounties

Earn bounties for contributing to opendbc!

*   $2000 - [Any car brand / platform port](https://github.com/orgs/commaai/projects/26/views/1?pane=issue&itemId=47913774)
*   $250 - [Any car model port](https://github.com/orgs/commaai/projects/26/views/1?pane=issue&itemId=47913790)
*   $300 - [Reverse Engineering a new Actuation Message](https://github.com/orgs/commaai/projects/26/views/1?pane=issue&itemId=73445563)

More bounties are available at [comma.ai/bounties](comma.ai/bounties).

## FAQ

*   **How do I use this?** A [comma 3X](https://comma.ai/shop/comma-3x) is recommended for running opendbc.
*   **Which cars are supported?** See the [supported cars list](docs/CARS.md).
*   **Can I add support for my car?** Yes, read the guide [here](https://github.com/commaai/opendbc/blob/docs/README.md#how-to-port-a-car).
*   **Which cars can be supported?** Cars with LKAS and ACC. More info [here](https://github.com/commaai/openpilot/blob/master/docs/CARS.md#dont-see-your-car-here).
*   **How does this work?** Hardware replaces your car's built-in lane keep and adaptive cruise features. See [this talk](https://www.youtube.com/watch?v=FL8CxUSfipM).
*   **Is there a timeline or roadmap for adding car support?** Community-driven, validated by comma.

### Terms

*   **port**: Car integration.
*   **lateral control**: Steering.
*   **longitudinal control**: Gas/brakes.
*   **fingerprinting**: Car identification.
*   **[LKAS](https://en.wikipedia.org/wiki/Lane_departure_warning_system)**: Lane keeping assist.
*   **[ACC](https://en.wikipedia.org/wiki/Adaptive_cruise_control)**: Adaptive cruise control.
*   **[harness](https://comma.ai/shop/car-harness)**: ADAS message interface.
*   **[panda](https://github.com/commaai/panda)**: CAN bus access hardware.
*   **[ECU](https://en.wikipedia.org/wiki/Electronic_control_unit)**: Car computers.
*   **[CAN bus](https://en.wikipedia.org/wiki/CAN_bus)**: ECU communication bus.
*   **[cabana](https://github.com/commaai/openpilot/tree/master/tools/cabana#readme)**: Reverse engineering tool.
*   **[DBC file](https://en.wikipedia.org/wiki/CAN_bus#DBC)**: CAN message definitions.
*   **[openpilot](https://github.com/commaai/openpilot)**: ADAS system using opendbc.
*   **[comma](https://github.com/commaai)**: Company behind opendbc.
*   **[comma 3X](https://comma.ai/shop/comma-3x)**: Hardware for running openpilot.

### More resources

*   [*How Do We Control The Car?*](https://www.youtube.com/watch?v=nNU6ipme878&pp=ygUoY29tbWEgY29uIDIwMjEgaG93IGRvIHdlIGNvbnRyb2wgdGhlIGNhcg%3D%3D) by [@robbederks](https://github.com/robbederks) from COMMA_CON 2021
*   [*How to Port a Car*](https://www.youtube.com/watch?v=XxPS5TpTUnI&t=142s&pp=ygUPamFzb24gY29tbWEgY29u) by [@jyoung8607](https://github.com/jyoung8607) from COMMA_CON 2023
*   [commaCarSegments](https://huggingface.co/datasets/commaai/commaCarSegments): a massive dataset of CAN data from 300 different car models
*   [cabana](https://github.com/commaai/openpilot/tree/master/tools/cabana#readme): our tool for reverse engineering CAN messages
*   [can_print_changes.py](https://github.com/commaai/openpilot/blob/master/selfdrive/debug/can_print_changes.py): diff the whole CAN bus across two drives, such as one without any LKAS and one with LKAS
*   [longitudinal maneuvers](https://github.com/commaai/openpilot/tree/master/tools/longitudinal_maneuvers): a tool for evaluating and tuning longitudinal control
*   [opendbc data](https://commaai.github.io/opendbc-data/): a repository of longitudinal maneuver evaluations

## Join the Team: [comma.ai/jobs](https://comma.ai/jobs)

comma is hiring engineers to work on opendbc and [openpilot](https://github.com/commaai/openpilot), and welcomes contributions!