<div align="center">

<h1>opendbc: Unlock Your Car's Potential with Python</h1>
<p>
  <b>Control your car's steering, gas, brakes, and more with opendbc, a powerful Python API.</b>
  <br>
  Read vehicle data, build custom applications, and contribute to the future of automotive technology.
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

##  opendbc: Your Gateway to Automotive Control

opendbc is a Python API that lets you interact with your car's systems, including steering, acceleration, and braking. Built with a focus on supporting the development of advanced driver-assistance systems (ADAS), opendbc empowers you to build custom vehicle applications and unlock new possibilities.

**Key Features:**

*   **Precise Control:** Directly command steering, gas, and brakes.
*   **Real-time Data Access:** Read crucial vehicle information like speed and steering angle.
*   **Comprehensive Car Support:** Aims to support control of steering, gas, and brakes on a wide range of vehicles.
*   **Community-Driven:** Benefit from a thriving community and contribute to expanding car compatibility.
*   **ADAS Integration:** Designed to integrate with openpilot, providing a solid foundation for your autonomous driving projects.
*   **Safety-Focused:** Includes robust safety features and rigorous code standards.

**Get Started Quickly:**

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/commaai/opendbc.git
    cd opendbc
    ```
2.  **Install Dependencies and Build:**
    ```bash
    ./test.sh
    ```

    This command installs dependencies, builds the project, runs tests, and performs linting.

**Explore Examples:**
Find example programs in the [`examples/`](examples/) directory to help you get started, including [`examples/joystick.py`](examples/joystick.py) for joystick control.

## Project Structure

*   [`opendbc/dbc/`](opendbc/dbc/) Contains DBC files, acting as databases for CAN message definitions.
*   [`opendbc/can/`](opendbc/can/) Provides a library to parse and build CAN messages using DBC files.
*   [`opendbc/car/`](opendbc/car/) Contains high-level Python library for interfacing with various car models.
*   [`opendbc/safety/`](opendbc/safety/) Implements functional safety mechanisms.

## How to Add Support for Your Car

This project is community-driven and welcomes contributions to add support for new car models. Detailed instructions are available to guide you through the process of supporting new vehicles, including guidance on:

*   Connecting to your car.
*   Understanding the structure of a car port.
*   Reverse engineering CAN messages.
*   Tuning for optimal performance.

## Contributing

We welcome contributions! Coordinate with the community on [GitHub](https://github.com/commaai/opendbc) and [Discord](https://discord.comma.ai). Check out the `#dev-opendbc-cars` channel and `Vehicle Specific` section for discussions.

**Roadmap:**

*   [ ] `pip install opendbc`
*   [ ] 100% type coverage
*   [ ] 100% line coverage
*   [ ] Make car ports easier: refactors, tools, tests, and docs
*   [ ] Expose the state of all supported cars better: https://github.com/commaai/opendbc/issues/1144
*   [ ] Extend support to every car with LKAS + ACC interfaces
*   [ ] Automatic lateral and longitudinal control/tuning evaluation
*   [ ] Auto-tuning for lateral and longitudinal control
*   [ ] Automatic Emergency Braking

## Safety Model

The opendbc safety firmware is written to maintain the [openpilot safety](https://github.com/commaai/openpilot/blob/master/docs/SAFETY.md). Rigorous testing and code standards are applied.

**Safety Features:**

*   `SAFETY_SILENT` mode by default.
*   `controls_allowed` for advanced safety control
*   Static code analysis with [cppcheck](https://github.com/danmar/cppcheck/).
*   [MISRA C:2012](https://misra.org.uk/) compliance.
*   Strict compiler flags.
*   Unit tests with 100% line coverage.

## Bounties

Earn rewards for contributing! Bounty programs are available for:

*   Adding support for a new car brand/platform.
*   Adding support for a new car model.
*   Reverse engineering a new actuation message.
*   Bounties for more popular cars are available at [comma.ai/bounties](comma.ai/bounties).

## FAQ

*   **How do I use this?** A [comma 3X](https://comma.ai/shop/comma-3x) is recommended for development.
*   **Which cars are supported?**  See the [supported cars list](docs/CARS.md).
*   **Can I add support for my car?** Yes, read the guide [here](https://github.com/commaai/opendbc/blob/docs/README.md#how-to-port-a-car).
*   **Which cars can be supported?** Cars with LKAS and ACC. More info [here](https://github.com/commaai/openpilot/blob/master/docs/CARS.md#dont-see-your-car-here).
*   **How does this work?**  It replaces your car's lane keep and adaptive cruise features. See [this talk](https://www.youtube.com/watch?v=FL8CxUSfipM).
*   **Is there a timeline or roadmap for adding car support?** Community-driven.

### Terms

*   **port**: Support for a specific car
*   **lateral control**: Steering control
*   **longitudinal control**: Gas/brakes control
*   **fingerprinting**: Identifying the car
*   **[LKAS](https://en.wikipedia.org/wiki/Lane_departure_warning_system)**: Lane keeping assist
*   **[ACC](https://en.wikipedia.org/wiki/Adaptive_cruise_control)**: Adaptive cruise control
*   **[harness](https://comma.ai/shop/car-harness)**: Car-specific hardware
*   **[panda](https://github.com/commaai/panda)**: Hardware for CAN bus access
*   **[ECU](https://en.wikipedia.org/wiki/Electronic_control_unit)**: Control modules
*   **[CAN bus](https://en.wikipedia.org/wiki/CAN_bus)**: Car's communication bus
*   **[cabana](https://github.com/commaai/openpilot/tree/master/tools/cabana#readme)**: Tool for reverse engineering CAN messages
*   **[DBC file](https://en.wikipedia.org/wiki/CAN_bus#DBC)**: Contains CAN message definitions
*   **[openpilot](https://github.com/commaai/openpilot)**: ADAS system
*   **[comma](https://github.com/commaai)**: The company
*   **[comma 3X](https://comma.ai/shop/comma-3x)**: The recommended hardware

### Additional Resources

*   [*How Do We Control The Car?*](https://www.youtube.com/watch?v=nNU6ipme878&pp=ygUoY29tbWEgY29uIDIwMjEgaG93IGRvIHdlIGNvbnRyb2wgdGhlIGNhcg%3D%3D)
*   [*How to Port a Car*](https://www.youtube.com/watch?v=XxPS5TpTUnI&t=142s&pp=ygUPamFzb24gY29tbWEgY29u)
*   [commaCarSegments](https://huggingface.co/datasets/commaai/commaCarSegments)
*   [cabana](https://github.com/commaai/openpilot/tree/master/tools/cabana#readme)
*   [can_print_changes.py](https://github.com/commaai/openpilot/blob/master/selfdrive/debug/can_print_changes.py)
*   [longitudinal maneuvers](https://github.com/commaai/openpilot/tree/master/tools/longitudinal_maneuvers)
*   [opendbc data](https://commaai.github.io/opendbc-data/)

## Join the Team

[comma.ai/jobs](https://comma.ai/jobs) is hiring engineers to work on opendbc and [openpilot](https://github.com/commaai/openpilot).