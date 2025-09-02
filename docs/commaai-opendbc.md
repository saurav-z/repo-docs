<div align="center" style="text-align: center;">

<h1>opendbc: Your Python API for Car Control and Data</h1>
<p>
  <b>Take control of your vehicle's systems with opendbc, empowering you to control and read data from your car.</b>
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

opendbc is a powerful Python API that allows you to interact with your car's systems, giving you the ability to control steering, gas, brakes, and access valuable data like speed and steering angle. Built to support ADAS interfaces, this project aims to provide comprehensive vehicle management capabilities. For more information, visit the [original repository](https://github.com/commaai/opendbc).

**Key Features:**

*   **Control & Read Data:**  Access and manipulate your car's core functions through a Python API.
*   **ADAS Integration:** Designed to support and enhance [openpilot](https://github.com/commaai/openpilot) features.
*   **Extensive Car Support:**  With the goal of expanding compatibility to a wide range of vehicles equipped with LKAS and ACC.
*   **Open Source & Community Driven:** A collaborative project with active contributions and community support.
*   **Safety Focused:** Implements a robust safety model for reliable operation.

---

## Getting Started

Follow these steps to get up and running quickly:

```bash
git clone https://github.com/commaai/opendbc.git
cd opendbc

# Install dependencies, build, run tests, and lint (recommended)
./test.sh

# Individual commands for advanced use
pip3 install -e .[testing,docs]  # Install dependencies
scons -j8                        # Build with 8 cores
pytest .                         # Run tests
lefthook run lint                # Run linter
```

Explore the [`examples/`](examples/) directory for sample programs.  [`examples/joystick.py`](examples/joystick.py) provides an example of controlling a car with a joystick.

## Project Structure

*   `opendbc/dbc/`: Contains DBC (Database CAN) files for different car models.
*   `opendbc/can/`:  Provides a library for parsing and creating CAN messages from DBC files.
*   `opendbc/car/`: High-level Python library for interacting with vehicle systems.
*   `opendbc/safety/`:  Implements functional safety mechanisms for supported cars.

## Contributing and Car Porting

opendbc thrives on community contributions. Detailed guides and resources are available for those who wish to add support for new vehicles or improve existing implementations.

*   **How to Port a Car:** Detailed instructions are available in the README.
*   **Contributing:** Get involved by participating on GitHub and [Discord](https://discord.comma.ai).
*   **Roadmap:** See current and future project goals in the Roadmap section.

### Code Rigor

opendbc prioritizes code quality and safety:

*   Static code analysis with [cppcheck](https://github.com/danmar/cppcheck/).
*   MISRA C:2012 compliance checks.
*   Strict compiler flags for rigorous code validation.
*   Extensive unit tests with mutation testing and coverage enforcement.
*   Ruff linter and mypy integration.

## Bounties

Contribute and earn bounties!

*   $2000 - [Any car brand / platform port](https://github.com/orgs/commaai/projects/26/views/1?pane=issue&itemId=47913774)
*   $250 - [Any car model port](https://github.com/orgs/commaai/projects/26/views/1?pane=issue&itemId=47913790)
*   $300 - [Reverse Engineering a new Actuation Message](https://github.com/orgs/commaai/projects/26/views/1?pane=issue&itemId=73445563)

Additional bounties are offered for popular car models.

## Frequently Asked Questions (FAQ)

*   **How do I use this?**  A [comma 3X](https://comma.ai/shop/comma-3x) is recommended for development.
*   **Which cars are supported?**  See the [supported cars list](docs/CARS.md).
*   **Can I add support for my car?** Yes, see the [car porting guide](https://github.com/commaai/opendbc/blob/docs/README.md#how-to-port-a-car).
*   **Which cars can be supported?**  Any car with LKAS and ACC features.
*   **How does this work?**  Hardware replaces built-in lane keep and adaptive cruise features. See the [in-depth explanation](https://www.youtube.com/watch?v=FL8CxUSfipM).
*   **Is there a timeline for adding car support?**  Community-driven, with comma validating ports.

### Terms

*   **port**: Car integration and support.
*   **lateral control**: Steering control.
*   **longitudinal control**: Gas/brake control.
*   **fingerprinting**: Automatic car identification.
*   **LKAS**: Lane Keeping Assist.
*   **ACC**: Adaptive Cruise Control.
*   **Harness**: Car-specific hardware.
*   **Panda**: Hardware for CAN bus access.
*   **ECU**: Electronic Control Unit.
*   **CAN bus**: Car communication network.
*   **Cabana**: Tool for reverse engineering CAN messages.
*   **DBC file**: Defines CAN bus messages.
*   **Openpilot**:  ADAS system.
*   **Comma**:  The company behind opendbc.
*   **Comma 3X**: Hardware for running openpilot and opendbc.

### More Resources

*   [*How Do We Control The Car?*](https://www.youtube.com/watch?v=nNU6ipme878&pp=ygUoY29tbWEgY29uIDIwMjEgaG93IGRvIHdlIGNvbnRyb2wgdGhlIGNhcg%3D%3D) by [@robbederks](https://github.com/robbederks) from COMMA_CON 2021
*   [*How to Port a Car*](https://www.youtube.com/watch?v=XxPS5TpTUnI&t=142s&pp=ygUPamFzb24gY29tbWEgY29u) by [@jyoung8607](https://github.com/jyoung8607) from COMMA_CON 2023
*   [commaCarSegments](https://huggingface.co/datasets/commaai/commaCarSegments): a massive dataset of CAN data from 300 different car models
*   [cabana](https://github.com/commaai/openpilot/tree/master/tools/cabana#readme): our tool for reverse engineering CAN messages
*   [can_print_changes.py](https://github.com/commaai/openpilot/blob/master/selfdrive/debug/can_print_changes.py): diff the whole CAN bus across two drives, such as one without any LKAS and one with LKAS
*   [longitudinal maneuvers](https://github.com/commaai/openpilot/tree/master/tools/longitudinal_maneuvers): a tool for evaluating and tuning longitudinal control
*   [opendbc data](https://commaai.github.io/opendbc-data/): a repository of longitudinal maneuver evaluations

## Join the Team - [comma.ai/jobs](https://comma.ai/jobs)

comma is actively seeking engineers to work on opendbc and [openpilot](https://github.com/commaai/openpilot).  We encourage community contributions.