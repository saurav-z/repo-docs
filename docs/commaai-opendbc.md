<div align="center" style="text-align: center;">

<h1>opendbc: Open Source Car Control and Data API</h1>
<p>
  <b>Take control of your car's steering, gas, and brakes with opendbc, a powerful Python API.</b>
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

**opendbc** is a comprehensive Python API designed for interacting with your car's systems. This open-source project allows you to control steering, acceleration, braking, and read real-time data from your vehicle, making it a valuable tool for developers, researchers, and automotive enthusiasts. Built primarily to support ADAS interfaces for [openpilot](https://github.com/commaai/openpilot), opendbc also aims to provide read/write capabilities for a wide range of car functions.

**[View the opendbc repository on GitHub](https://github.com/commaai/opendbc)**

## Key Features

*   **Control:** Directly manipulate steering, gas, and brakes.
*   **Data Acquisition:** Read crucial vehicle data like speed and steering angle.
*   **Open Source:** Contribute to the project and expand its capabilities.
*   **Car Compatibility:** Support for a growing list of vehicles with LKAS and ACC.
*   **Community Driven:** Benefit from the collective knowledge and contributions of the open-source community.
*   **Extensible:**  Easily extend support to new cars and features.

## Getting Started

Get up and running quickly with the following commands:

```bash
git clone https://github.com/commaai/opendbc.git
cd opendbc

# Install dependencies, build, and run tests with a single command (recommended)
./test.sh

# Alternatively, run these commands individually:
pip3 install -e .[testing,docs]  # Install dependencies
scons -j8                        # Build with 8 cores
pytest .                         # Run the tests
lefthook run lint                # Run the linter
```

Explore example programs to get familiar with the API:

*   [`examples/`](examples/) contains simple programs to read and control car functions.
*   [`examples/joystick.py`](examples/joystick.py) lets you control a car using a joystick.

### Project Structure

*   [`opendbc/dbc/`](opendbc/dbc/) : Stores [DBC](https://en.wikipedia.org/wiki/CAN_bus#DBC_(CAN_Database_Files)) files for various cars.
*   [`opendbc/can/`](opendbc/can/) : Provides a library for parsing and constructing CAN messages using DBC files.
*   [`opendbc/car/`](opendbc/car/) : Contains a high-level Python library for interacting with cars.
*   [`opendbc/safety/`](opendbc/safety/) : Houses the functional safety mechanisms for all supported cars.

## Contributing to opendbc

Development happens on GitHub and the [Discord](https://discord.comma.ai) server. Check out the `#dev-opendbc-cars` channel and `Vehicle Specific` section.

### How to Port a Car

The goal of this project is to support controlling the steering, gas, and brakes on every single one of those cars.

Follow this guide to add support for a new car or improve existing ones:

1.  **Connect to the Car:** You'll need a comma 3X and a compatible car harness.
2.  **Structure of a port:** The entirety of a car port lives in `opendbc/car/<brand>/`. This includes `carstate.py`, `carcontroller.py`, `<brand>can.py`, `fingerprints.py`, `interface.py`, `radar_interface.py`, and `values.py`.
3.  **Reverse Engineer CAN messages:**  Use [cabana](https://github.com/commaai/openpilot/tree/master/tools/cabana) to analyze CAN messages.
4.  **Tuning:** Refine longitudinal control using the [longitudinal maneuvers](https://github.com/commaai/openpilot/tree/master/tools/longitudinal_maneuvers) tool.

### Roadmap

Short term
- [ ] `pip install opendbc`
- [ ] 100% type coverage
- [ ] 100% line coverage
- [ ] Make car ports easier: refactors, tools, tests, and docs
- [ ] Expose the state of all supported cars better: https://github.com/commaai/opendbc/issues/1144

Longer term
- [ ] Extend support to every car with LKAS + ACC interfaces
- [ ] Automatic lateral and longitudinal control/tuning evaluation
- [ ] Auto-tuning for [lateral](https://blog.comma.ai/090release/#torqued-an-auto-tuner-for-lateral-control) and longitudinal control
- [ ] [Automatic Emergency Braking](https://en.wikipedia.org/wiki/Automated_emergency_braking_system)

All contributions are welcome!

## Safety Model & Code Rigor

The opendbc safety firmware, integrated with [openpilot](https://github.com/commaai/openpilot) and [panda](https://github.com/commaai/panda), ensures the safety of the system. It adheres to stringent coding standards to maintain code quality. The safety firmware is written for its use in conjunction with [openpilot](https://github.com/commaai/openpilot) and [panda](https://github.com/commaai/panda). The safety firmware, through its safety model, provides and enforces the
[openpilot safety](https://github.com/commaai/openpilot/blob/master/docs/SAFETY.md). Due to its critical function, it's important that the application code rigor within the `safety` folder is held to high standards.

Code rigor includes:

*   Static code analysis with [cppcheck](https://github.com/danmar/cppcheck/), including [MISRA C:2012](https://misra.org.uk/) checks.
*   Strict compiler options (`-Wall -Wextra -Wstrict-prototypes -Werror`).
*   Extensive [unit tests](opendbc/safety/tests), including mutation tests on the MISRA coverage, and 100% line coverage.
*   Ruff linter and mypy for the car interface library.

## Bounties

*   $2000 - [Any car brand / platform port](https://github.com/orgs/commaai/projects/26/views/1?pane=issue&itemId=47913774)
*   $250 - [Any car model port](https://github.com/orgs/commaai/projects/26/views/1?pane=issue&itemId=47913790)
*   $300 - [Reverse Engineering a new Actuation Message](https://github.com/orgs/commaai/projects/26/views/1?pane=issue&itemId=73445563)

Additional bounties are available for popular cars at [comma.ai/bounties](comma.ai/bounties).

## Frequently Asked Questions (FAQ)

**Q: How do I use this?**
A: The [comma 3X](https://comma.ai/shop/comma-3x) is the recommended hardware for development and operation of opendbc and openpilot.

**Q: Which cars are supported?**
A: Check the [supported cars list](docs/CARS.md).

**Q: Can I add support for my car?**
A: Yes!  The community drives much of the car support. Follow the [porting guide](https://github.com/commaai/opendbc/blob/docs/README.md#how-to-port-a-car).

**Q: Which cars can be supported?**
A: Any car with LKAS and ACC. See more info [here](https://github.com/commaai/openpilot/blob/master/docs/CARS.md#dont-see-your-car-here).

**Q: How does this work?**
A: We designed hardware to replace the car's lane keep and adaptive cruise control features.  See [this talk](https://www.youtube.com/watch?v=FL8CxUSfipM) for an explanation.

**Q: Is there a timeline or roadmap for adding car support?**
A:  Community contributions drive car support, with comma performing final safety validation.

### Definitions

*   **port**: Implementing support for a specific car.
*   **lateral control**: Steering control.
*   **longitudinal control**: Gas/brake control.
*   **fingerprinting**: Automatic car identification.
*   **[LKAS](https://en.wikipedia.org/wiki/Lane_departure_warning_system)**: Lane Keeping Assist.
*   **[ACC](https://en.wikipedia.org/wiki/Adaptive_cruise_control)**: Adaptive Cruise Control.
*   **[harness](https://comma.ai/shop/car-harness)**: Car-specific hardware for connecting to the car's ADAS systems.
*   **[panda](https://github.com/commaai/panda)**: Hardware for accessing a car's CAN bus.
*   **[ECU](https://en.wikipedia.org/wiki/Electronic_control_unit)**: Electronic Control Unit, a computer in a car.
*   **[CAN bus](https://en.wikipedia.org/wiki/CAN_bus)**: The network that connects ECUs in a car.
*   **[cabana](https://github.com/commaai/openpilot/tree/master/tools/cabana#readme)**: Tool for reverse engineering CAN messages.
*   **[DBC file](https://en.wikipedia.org/wiki/CAN_bus#DBC)**: Defines CAN message structures.
*   **[openpilot](https://github.com/commaai/openpilot)**: ADAS system compatible with opendbc.
*   **[comma](https://github.com/commaai)**: The company behind opendbc.
*   **[comma 3X](https://comma.ai/shop/comma-3x)**: Hardware for running openpilot.

### Additional Resources

*   [*How Do We Control The Car?*](https://www.youtube.com/watch?v=nNU6ipme878&pp=ygUoY29tbWEgY29uIDIwMjEgaG93IGRvIHdlIGNvbnRyb2wgdGhlIGNhcg%3D%3D) by [@robbederks](https://github.com/robbederks) from COMMA_CON 2021
*   [*How to Port a Car*](https://www.youtube.com/watch?v=XxPS5TpTUnI&t=142s&pp=ygUPamFzb24gY29tbWEgY29u) by [@jyoung8607](https://github.com/jyoung8607) from COMMA_CON 2023
*   [commaCarSegments](https://huggingface.co/datasets/commaai/commaCarSegments): a massive dataset of CAN data from 300 different car models
*   [cabana](https://github.com/commaai/openpilot/tree/master/tools/cabana#readme): our tool for reverse engineering CAN messages
*   [can_print_changes.py](https://github.com/commaai/openpilot/blob/master/selfdrive/debug/can_print_changes.py): diff the whole CAN bus across two drives, such as one without any LKAS and one with LKAS
*   [longitudinal maneuvers](https://github.com/commaai/openpilot/tree/master/tools/longitudinal_maneuvers): a tool for evaluating and tuning longitudinal control
*   [opendbc data](https://commaai.github.io/opendbc-data/): a repository of longitudinal maneuver evaluations

## Join the Team - [comma.ai/jobs](https://comma.ai/jobs)

comma is hiring engineers to work on opendbc and [openpilot](https://github.com/commaai/openpilot). We love hiring contributors.