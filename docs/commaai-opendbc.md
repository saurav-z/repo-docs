# opendbc: Your Python API for Vehicle Control and Data

**opendbc** empowers you to control, read data from, and extend the capabilities of your car's systems.  Access your car's steering, gas, brakes, and more with this powerful Python API.  [Explore the opendbc repository on GitHub](https://github.com/commaai/opendbc).

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![X Follow](https://img.shields.io/twitter/follow/comma_ai)](https://x.com/comma_ai)
[![Discord](https://img.shields.io/discord/469524606043160576)](https://discord.comma.ai)

## Key Features

*   **Control & Read Data:** Access and manipulate vehicle functions like steering, acceleration, braking, and gather real-time data such as speed and steering angle.
*   **Broad Vehicle Support:**  Designed to support a wide range of vehicles with electronically-actuated systems, including those with LKAS and ACC.
*   **Easy Integration:** Leverage a Python API for seamless integration into your projects.
*   **Community Driven:** Benefit from a collaborative, open-source project with active community contributions.
*   **Extensive Documentation:** Comprehensive documentation is available within the repository and related resources.
*   **Safety Focused:** Built-in safety features and rigorous code rigor to ensure reliable and safe operation.

## Getting Started

### Installation

```bash
git clone https://github.com/commaai/opendbc.git
cd opendbc

# Comprehensive setup for dependencies, building, linting, and testing.
./test.sh

# Individual commands
pip3 install -e .[testing,docs]  # Install dependencies
scons -j8                        # Build with 8 cores
pytest .                         # Run the tests
lefthook run lint                # Run the linter
```

### Examples

Explore the `examples/` directory for practical code snippets demonstrating how to read car state and control various systems.  Experiment with `examples/joystick.py` to control your car using a joystick.

## Project Structure

*   **`opendbc/dbc/`**: Contains DBC files, which define CAN bus message formats.
*   **`opendbc/can/`**:  A library for parsing and constructing CAN messages using DBC files.
*   **`opendbc/car/`**: A high-level Python library providing a user-friendly interface for interacting with vehicles.
*   **`opendbc/safety/`**: Houses the safety-critical components and firmware for the supported vehicles.

## Porting a Car: Expand Vehicle Support

The opendbc project welcomes contributions to expand support for a wider range of vehicles. Learn how to contribute by reading the [How to Port a Car](https://github.com/commaai/opendbc/blob/docs/README.md#how-to-port-a-car) guide.

### Key Steps:

1.  **Connect to the Car:** Establish a connection to the vehicle's CAN buses using a comma 3X and car harness.
2.  **Understand the Structure:**  Familiarize yourself with the structure of a car port, including `carstate.py`, `carcontroller.py`, and related files.
3.  **Reverse Engineer CAN Messages:**  Use tools like [cabana](https://github.com/commaai/openpilot/tree/master/tools/cabana) to analyze CAN bus data.
4.  **Tuning and Testing:**  Utilize the provided tools and resources to tune and test your car port.

## Contributing

Get involved with opendbc! All development is coordinated on GitHub and the [Discord](https://discord.comma.ai) server. Join the `#dev-opendbc-cars` channel and the `Vehicle Specific` section to collaborate with the community.

### Roadmap

*   [ ] `pip install opendbc`
*   [ ] 100% type coverage
*   [ ] 100% line coverage
*   [ ] Make car ports easier: refactors, tools, tests, and docs
*   [ ] Expose the state of all supported cars better: https://github.com/commaai/opendbc/issues/1144

**Future Goals:** Expand support to every car with LKAS + ACC interfaces,  enhance control with automatic tuning, and provide support for Automatic Emergency Braking (AEB).

## Safety Model

The opendbc safety firmware, critical for use with [openpilot](https://github.com/commaai/openpilot), operates in `SAFETY_SILENT` mode by default.  Select a safety mode to send messages.

## Code Rigor

The safety firmware undergoes rigorous testing and adheres to high coding standards, including static code analysis, MISRA C:2012 compliance, strict compiler flags, and comprehensive unit tests.

## Bounties

Contribute to opendbc and earn bounties!

*   $2000 - [Any car brand / platform port](https://github.com/orgs/commaai/projects/26/views/1?pane=issue&itemId=47913774)
*   $250 - [Any car model port](https://github.com/orgs/commaai/projects/26/views/1?pane=issue&itemId=47913790)
*   $300 - [Reverse Engineering a new Actuation Message](https://github.com/orgs/commaai/projects/26/views/1?pane=issue&itemId=73445563)

Explore bounties for popular car models at [comma.ai/bounties](https://comma.ai/bounties).

## Frequently Asked Questions

*   **How do I use this?** A [comma 3X](https://comma.ai/shop/comma-3x) is recommended to use and develop opendbc and openpilot.
*   **Which cars are supported?** Check the [supported cars list](docs/CARS.md).
*   **Can I add support for my car?** Yes, community contributions are welcome; see the [car porting guide](https://github.com/commaai/opendbc/blob/docs/README.md#how-to-port-a-car).
*   **Which cars can be supported?** Any car with LKAS and ACC. More info [here](https://github.com/commaai/openpilot/blob/master/docs/CARS.md#dont-see-your-car-here).
*   **How does this work?** We designed hardware to replace your car's built-in lane keep and adaptive cruise features. See [this talk](https://www.youtube.com/watch?v=FL8CxUSfipM) for an in-depth explanation.
*   **Is there a timeline or roadmap for adding car support?** Most car support comes from the community, with comma doing final safety and quality validation.

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

## More Resources

*   [*How Do We Control The Car?*](https://www.youtube.com/watch?v=nNU6ipme878&pp=ygUoY29tbWEgY29uIDIwMjEgaG93IGRvIHdlIGNvbnRyb2wgdGhlIGNhcg%3D%3D) by [@robbederks](https://github.com/robbederks) from COMMA_CON 2021
*   [*How to Port a Car*](https://www.youtube.com/watch?v=XxPS5TpTUnI&t=142s&pp=ygUPamFzb24gY29tbWEgY29u) by [@jyoung8607](https://github.com/jyoung8607) from COMMA_CON 2023
*   [commaCarSegments](https://huggingface.co/datasets/commaai/commaCarSegments): a massive dataset of CAN data from 300 different car models
*   [cabana](https://github.com/commaai/openpilot/tree/master/tools/cabana#readme): our tool for reverse engineering CAN messages
*   [can_print_changes.py](https://github.com/commaai/openpilot/blob/master/selfdrive/debug/can_print_changes.py): diff the whole CAN bus across two drives, such as one without any LKAS and one with LKAS
*   [longitudinal maneuvers](https://github.com/commaai/openpilot/tree/master/tools/longitudinal_maneuvers): a tool for evaluating and tuning longitudinal control
*   [opendbc data](https://commaai.github.io/opendbc-data/): a repository of longitudinal maneuver evaluations

## Join the Team: [comma.ai/jobs](https://comma.ai/jobs)

comma is actively hiring engineers to work on opendbc and [openpilot](https://github.com/commaai/openpilot). Contributors are encouraged to apply.