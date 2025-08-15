# opendbc: Python API for Your Car

**Control, read, and understand your vehicle's data with opendbc, a powerful Python API for interacting with your car's CAN bus.**  ([Original Repository](https://github.com/commaai/opendbc))

[<img src="https://img.shields.io/github/stars/commaai/opendbc?style=social" alt="GitHub Stars">](https://github.com/commaai/opendbc)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![X Follow](https://img.shields.io/twitter/follow/comma_ai)](https://x.com/comma_ai)
[![Discord](https://img.shields.io/discord/469524606043160576)](https://discord.comma.ai)

## Key Features

*   **Comprehensive Car Control:** Control steering, gas, brakes, and more.
*   **Real-time Data Access:** Read speed, steering angle, and other vital vehicle data.
*   **DBC File Integration:** Uses DBC files for parsing and building CAN messages.
*   **Extensive Car Support:** Designed to support a wide range of vehicles with LKAS and ACC.
*   **Open Source & Community Driven:** Benefit from a collaborative community and contribute to its development.
*   **Safety Focused:** Rigorous testing and safety models for reliable performance.

## Core Functionality

opendbc provides a Python-based interface to interact with your car's Controller Area Network (CAN) bus. It allows you to both read data from your car (e.g., speed, steering angle) and send control commands (e.g., steering, gas, brakes).

### Quick Start

```bash
git clone https://github.com/commaai/opendbc.git
cd opendbc

# Install dependencies, build, and run tests (recommended)
./test.sh

# Individual commands for more control
pip3 install -e .[testing,docs]  # install dependencies
scons -j8                        # build with 8 cores
pytest .                         # run the tests
lefthook run lint                # run the linter
```

Example programs within the [`examples/`](examples/) directory demonstrate reading vehicle state and controlling steering, gas, and brakes.  For example, [`examples/joystick.py`](examples/joystick.py) enables car control via a joystick.

## Project Structure

*   [`opendbc/dbc/`](opendbc/dbc/):  Stores DBC files that define CAN message structures.
*   [`opendbc/can/`](opendbc/can/): Provides a library for parsing and building CAN messages using DBC files.
*   [`opendbc/car/`](opendbc/car/): Offers a high-level Python library for interfacing with cars.
*   [`opendbc/safety/`](opendbc/safety/): Implements functional safety measures for supported vehicles.

## How to Port a Car

Expand the capabilities of opendbc by adding support for your car.  The process involves connecting to your car's CAN bus, reverse engineering CAN messages, and writing the necessary Python code.

1.  **Connect to the Car:** Use a comma 3X and a car harness.  Harnesses are available on comma.ai/shop.
2.  **Understand the Structure:** Car ports are organized within `opendbc/car/<brand>/`. Key files include:
    *   `carstate.py`: Parses CAN data.
    *   `carcontroller.py`: Sends control messages.
    *   `<brand>can.py`:  Helper functions for building CAN messages.
    *   `fingerprints.py`: Identifies car models.
    *   `interface.py`: Provides a high-level interface.
    *   `radar_interface.py`: Parses radar data.
    *   `values.py`: Defines supported car models.
3.  **Reverse Engineer CAN Messages:** Use tools like [cabana](https://github.com/commaai/openpilot/tree/master/tools/cabana) to analyze CAN data and identify relevant messages.
4.  **Tuning:** Evaluate and tune longitudinal control using tools like the [longitudinal maneuvers](https://github.com/commaai/openpilot/tree/master/tools/longitudinal_maneuvers) report.

## Contributing

Contribute to opendbc's development on GitHub and [Discord](https://discord.comma.ai).  Engage in the `#dev-opendbc-cars` channel for discussions and support.

### Roadmap

**Short-Term Goals:**

*   Implement `pip install opendbc`.
*   Achieve 100% type and line coverage.
*   Improve car port creation through refactoring, tooling, and documentation.
*   Enhance state visualization for supported vehicles (issue #1144).

**Longer-Term Goals:**

*   Extend support to every car with LKAS + ACC.
*   Automate lateral and longitudinal control and tuning evaluation.
*   Implement auto-tuning for lateral and longitudinal control.
*   Develop Automatic Emergency Braking functionality.

## Safety Model

The opendbc safety firmware, operating alongside [openpilot](https://github.com/commaai/openpilot) and [panda](https://comma.ai/shop/panda), ensures safe vehicle control. It defaults to `SAFETY_SILENT` mode, requiring selection of a safety mode to send messages.  Certain modes are disabled in release firmwares.  The safety model optionally incorporates `controls_allowed`, enabling or disabling specific messages based on board state.

## Code Rigor

The `safety` folder's application code adheres to stringent standards:

*   Static code analysis with [cppcheck](https://github.com/danmar/cppcheck/), including [MISRA C:2012](https://misra.org.uk/) violations checks.
*   Strict compiler flags (`-Wall -Wextra -Wstrict-prototypes -Werror`).
*   Unit tests (100% line coverage) for safety logic for all supported car variants.
*   Mutation testing on MISRA coverage.
*   Ruff linter and mypy for the car interface library.

## Bounties

Contribute to opendbc and earn bounties:

*   $2000 -  [Any car brand / platform port](https://github.com/orgs/commaai/projects/26/views/1?pane=issue&itemId=47913774)
*   $250 -  [Any car model port](https://github.com/orgs/commaai/projects/26/views/1?pane=issue&itemId=47913790)
*   $300 - [Reverse Engineering a new Actuation Message](https://github.com/orgs/commaai/projects/26/views/1?pane=issue&itemId=73445563)

Higher-value bounties are available for popular cars. See [comma.ai/bounties](https://comma.ai/bounties) for the latest opportunities.

## FAQ

*   **How do I use this?** Use a [comma 3X](https://comma.ai/shop/comma-3x) for optimal development and operation.
*   **Which cars are supported?**  Refer to the [supported cars list](docs/CARS.md).
*   **Can I add support for my car?**  Yes, the community drives car support. See the [car port guide](https://github.com/commaai/opendbc/blob/docs/README.md#how-to-port-a-car).
*   **Which cars can be supported?**  Any car with LKAS and ACC. More information [here](https://github.com/commaai/openpilot/blob/master/docs/CARS.md#dont-see-your-car-here).
*   **How does this work?**  Hardware replaces built-in features. See [this talk](https://www.youtube.com/watch?v=FL8CxUSfipM) for details.
*   **Is there a car support roadmap?**  No, community contributions are key.

### Terms

*   **port**:  Integrating and supporting a specific car
*   **lateral control**:  Steering control
*   **longitudinal control**: Gas/brake control
*   **fingerprinting**: Automatic car identification
*   **[LKAS](https://en.wikipedia.org/wiki/Lane_departure_warning_system)**: Lane Keeping Assist
*   **[ACC](https://en.wikipedia.org/wiki/Adaptive_cruise_control)**: Adaptive Cruise Control
*   **[harness](https://comma.ai/shop/car-harness)**: Hardware to interface with the car
*   **[panda](https://github.com/commaai/panda)**:  Hardware for CAN bus access
*   **[ECU](https://en.wikipedia.org/wiki/Electronic_control_unit)**: Car's control modules
*   **[CAN bus](https://en.wikipedia.org/wiki/CAN_bus)**: Vehicle's internal communication network
*   **[cabana](https://github.com/commaai/openpilot/tree/master/tools/cabana#readme)**: Tool for reverse engineering CAN messages
*   **[DBC file](https://en.wikipedia.org/wiki/CAN_bus#DBC)**: Defines CAN message contents
*   **[openpilot](https://github.com/commaai/openpilot)**: ADAS system utilizing opendbc
*   **[comma](https://github.com/commaai)**: The company behind opendbc
*   **[comma 3X](https://comma.ai/shop/comma-3x)**: Hardware for running openpilot

### More Resources

*   [*How Do We Control The Car?*](https://www.youtube.com/watch?v=nNU6ipme878&pp=ygUoY29tbWEgY29uIDIwMjEgaG93IGRvIHdlIGNvbnRyb2wgdGhlIGNhcg%3D%3D) by [@robbederks](https://github.com/robbederks) from COMMA_CON 2021
*   [*How to Port a Car*](https://www.youtube.com/watch?v=XxPS5TpTUnI&t=142s&pp=ygUPamFzb24gY29tbWEgY29u) by [@jyoung8607](https://github.com/jyoung8607) from COMMA_CON 2023
*   [commaCarSegments](https://huggingface.co/datasets/commaai/commaCarSegments): Massive CAN data dataset
*   [cabana](https://github.com/commaai/openpilot/tree/master/tools/cabana#readme): CAN message reverse engineering tool
*   [can_print_changes.py](https://github.com/commaai/openpilot/blob/master/selfdrive/debug/can_print_changes.py): CAN bus diff tool
*   [longitudinal maneuvers](https://github.com/commaai/openpilot/tree/master/tools/longitudinal_maneuvers): Longitudinal control evaluation
*   [opendbc data](https://commaai.github.io/opendbc-data/): Longitudinal maneuver evaluations

## Join the Team!

comma is hiring engineers! Explore opportunities at [comma.ai/jobs](https://comma.ai/jobs). Contributions are valued!
```
Key improvements and SEO optimizations:

*   **Clear Title:**  "opendbc: Python API for Your Car" is concise and includes the core functionality.
*   **One-Sentence Hook:** The initial sentence directly addresses the user's need.
*   **Keyword-Rich Description:**  Uses relevant keywords (e.g., "Python API," "car control," "CAN bus").
*   **Well-Defined Sections:** Clear headings and subheadings for readability and SEO.
*   **Bulleted Key Features:** Highlights core benefits in a user-friendly format.
*   **Clear "Quick Start" Instructions:**  Easy-to-follow instructions.
*   **Targeted Headings:**  Uses headings like "How to Port a Car" and "Contributing" to guide users.
*   **Internal Linking:** References relevant files and sections within the project.
*   **External Linking:** Includes links to the original repository, relevant documentation, and external resources (e.g., Wikipedia articles, videos).
*   **Call to Action:** Encourages contribution and job applications.
*   **FAQ Section:** Addresses common user questions for improved clarity and SEO.
*   **Terms Section:** Defines key terms related to the project for clarity and discoverability.
*   **Concise Language:** Uses clear and direct language.
*   **Use of Shields:** Keeps the badges from original README.
*   **SEO-Friendly Structure:** Uses proper headings, bold text, and lists for better search engine parsing.
*   **Added a GitHub stars badge** For social proof.