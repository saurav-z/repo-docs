# opendbc: Your Car's Digital Interface for Advanced Driver-Assistance Systems (ADAS)

**opendbc provides a powerful Python API that gives you control over your car's gas, brakes, and steering, empowering developers to build advanced driver-assistance systems.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![X Follow](https://img.shields.io/twitter/follow/comma_ai)](https://x.com/comma_ai)
[![Discord](https://img.shields.io/discord/469524606043160576)](https://discord.comma.ai)

[View the opendbc repository on GitHub](https://github.com/commaai/opendbc)

## Key Features

*   **Car Control:** Directly control your car's steering, gas, and brakes.
*   **Data Access:** Read real-time data like speed, steering angle, and more.
*   **DBC File Integration:** Utilizes DBC files for parsing and building CAN messages.
*   **Extensive Car Support:** Designed to support a wide range of vehicles with LKAS and ACC.
*   **Open Source:** A community-driven project with active development and contributions.
*   **Safety Focused:** Includes a robust safety model and rigorous code quality checks.
*   **Developer Friendly:** Offers comprehensive documentation and examples to get you started quickly.

## Getting Started

### Installation and Testing

```bash
git clone https://github.com/commaai/opendbc.git
cd opendbc

# Install dependencies, build, lint, and run tests all in one go
./test.sh
```

For detailed instructions, see the full README and the [supported cars list](docs/CARS.md).

## Project Structure

*   `opendbc/dbc/`: Stores [DBC](https://en.wikipedia.org/wiki/CAN_bus#DBC) files.
*   `opendbc/can/`: Library for parsing and building CAN messages.
*   `opendbc/car/`: High-level Python library for interacting with cars.
*   `opendbc/safety/`: Functional safety components.

## Contributing

opendbc welcomes contributions! Development is coordinated on GitHub and [Discord](https://discord.comma.ai). Join the `#dev-opendbc-cars` channel for collaboration.

## Roadmap

*   \[ ] `pip install opendbc`
*   \[ ] 100% type coverage
*   \[ ] 100% line coverage
*   \[ ] Make car ports easier: refactors, tools, tests, and docs
*   \[ ] Expose the state of all supported cars better: https://github.com/commaai/opendbc/issues/1144

## Bounties

Earn bounties for contributing:

*   $2000 - [Any car brand / platform port](https://github.com/orgs/commaai/projects/26/views/1?pane=issue&itemId=47913774)
*   $250 - [Any car model port](https://github.com/orgs/commaai/projects/26/views/1?pane=issue&itemId=47913790)
*   $300 - [Reverse Engineering a new Actuation Message](https://github.com/orgs/commaai/projects/26/views/1?pane=issue&itemId=73445563)

Find more bounties at [comma.ai/bounties](comma.ai/bounties).

## FAQ

**Q: How do I use this?**

A: A [comma 3X](https://comma.ai/shop/comma-3x) is recommended to run and develop opendbc and openpilot.

**Q: Which cars are supported?**

A: See the [supported cars list](docs/CARS.md).

**Q: Can I add support for my car?**

A: Yes! See the guide [here](https://github.com/commaai/opendbc/blob/docs/README.md#how-to-port-a-car).

**Q: Which cars can be supported?**

A: Any car with LKAS and ACC. More info [here](https://github.com/commaai/openpilot/blob/master/docs/CARS.md#dont-see-your-car-here).

**Q: How does this work?**

A: It replaces your car's built-in lane keep and adaptive cruise features. See [this talk](https://www.youtube.com/watch?v=FL8CxUSfipM) for an in-depth explanation.

**Q: Is there a timeline or roadmap for adding car support?**

A: Most car support comes from the community, with comma doing final safety and quality validation. The more complete the community car port is and the more popular the car is, the more likely we are to pick it up as the next one to validate.

## Terms

*   **port**: refers to the integration and support of a specific car
*   **lateral control**: aka steering control
*   **longitudinal control**: aka gas/brakes control
*   **fingerprinting**: automatic process for identifying the car
*   **[LKAS](https://en.wikipedia.org/wiki/Lane_departure_warning_system)**: lane keeping assist
*   **[ACC](https://en.wikipedia.org/wiki/Adaptive_cruise_control)**: adaptive cruise control
*   **[harness](https://comma.ai/shop/car-harness)**: car-specific hardware to attach to the car and intercept the ADAS messages
*   **[panda](https://github.com/commaai/panda)**: hardware used to get on a car's CAN bus
*   **[ECU](https://en.wikipedia.org/wiki/Electronic_control_unit)**: computers or control modules inside the car
*   **[CAN bus](https://en.wikipedia.org/wiki/CAN_bus)**: a bus that connects the ECUs in a car
*   **[cabana](https://github.com/commaai/openpilot/tree/master/tools/cabana#readme)**: our tool for reverse engineering CAN messages
*   **[DBC file](https://en.wikipedia.org/wiki/CAN_bus#DBC)**: contains definitions for messages on a CAN bus
*   **[openpilot](https://github.com/commaai/openpilot)**: an ADAS system for cars supported by opendbc
*   **[comma](https://github.com/commaai)**: the company behind opendbc
*   **[comma 3X](https://comma.ai/shop/comma-3x)**: the hardware used to run openpilot

## More resources

*   [*How Do We Control The Car?*](https://www.youtube.com/watch?v=nNU6ipme878&pp=ygUoY29tbWEgY29uIDIwMjEgaG93IGRvIHdlIGNvbnRyb2wgdGhlIGNhcg%3D%3D) by [@robbederks](https://github.com/robbederks) from COMMA_CON 2021
*   [*How to Port a Car*](https://www.youtube.com/watch?v=XxPS5TpTUnI&t=142s&pp=ygUPamFzb24gY29tbWEgY29u) by [@jyoung8607](https://github.com/jyoung8607) from COMMA_CON 2023
*   [commaCarSegments](https://huggingface.co/datasets/commaai/commaCarSegments): a massive dataset of CAN data from 300 different car models
*   [cabana](https://github.com/commaai/openpilot/tree/master/tools/cabana#readme): our tool for reverse engineering CAN messages
*   [can_print_changes.py](https://github.com/commaai/openpilot/blob/master/selfdrive/debug/can_print_changes.py): diff the whole CAN bus across two drives, such as one without any LKAS and one with LKAS
*   [longitudinal maneuvers](https://github.com/commaai/openpilot/tree/master/tools/longitudinal_maneuvers): a tool for evaluating and tuning longitudinal control
*   [opendbc data](https://commaai.github.io/opendbc-data/): a repository of longitudinal maneuver evaluations

## Come work with us -- [comma.ai/jobs](https://comma.ai/jobs)

comma is hiring engineers to work on opendbc and [openpilot](https://github.com/commaai/openpilot). We love hiring contributors.