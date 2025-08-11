<div align="center">
  <h1>opendbc: Open Source Car Control and Data Interface</h1>
  <p>
    <b>Take control of your car!</b> opendbc is a powerful Python API enabling you to control gas, brakes, steering, and access vital vehicle data.
  </p>
  <p>
    <a href="https://github.com/commaai/opendbc">
      <img src="https://img.shields.io/badge/GitHub-Repo-blue?style=flat-square" alt="GitHub Repo">
    </a>
    <a href="https://discord.comma.ai">
      <img src="https://img.shields.io/discord/469524606043160576?style=flat-square&label=Discord" alt="Discord">
    </a>
    <a href="https://x.com/comma_ai">
      <img src="https://img.shields.io/twitter/follow/comma_ai?style=flat-square" alt="Twitter">
    </a>
  </p>
</div>

---

opendbc empowers you to interact with your vehicle's systems, opening doors to advanced driver-assistance systems (ADAS) and comprehensive vehicle management. This project focuses on providing a robust and accessible interface for controlling and reading data from modern vehicles.

**Key Features:**

*   **Control:** Manipulate steering, gas, and brakes.
*   **Data Access:** Read vehicle speed, steering angle, and more.
*   **Car Support:** Expanding support for vehicles with LKAS and ACC.
*   **Open Source:**  Contribute and customize to your needs.
*   **Python API:** Easy to use and integrate.

**Resources:**

*   [Documentation](https://docs.comma.ai)
*   [Contribute](https://github.com/commaai/openpilot/blob/master/docs/CONTRIBUTING.md)
*   [Discord](https://discord.comma.ai)
*   [Supported Cars List](docs/CARS.md)

## Quick Start

Get started with opendbc by cloning the repository and running the test script:

```bash
git clone https://github.com/commaai/opendbc.git
cd opendbc
./test.sh
```

Explore the `examples/` directory for example programs, including `examples/joystick.py` for joystick control.

## Project Structure

*   [`opendbc/dbc/`](opendbc/dbc/):  DBC file repository (CAN database files).
*   [`opendbc/can/`](opendbc/can/):  CAN message parsing and building library.
*   [`opendbc/car/`](opendbc/car/):  High-level Python library for car interaction.
*   [`opendbc/safety/`](opendbc/safety/): Functional safety module for supported cars.

## How to Port a Car

This section details the process of adding support for new cars or improving existing ones.
The process generally involves connecting to the car's CAN buses, reverse engineering CAN messages, and creating the necessary Python interfaces.

### Connect to the Car

Connect to the car using a comma 3X and a car harness.  Harnesses are available at comma.ai/shop.

### Structure of a Port

A car port within `opendbc/car/<brand>/` will generally include the following:

*   `carstate.py`: Parses CAN data.
*   `carcontroller.py`: Sends control CAN messages.
*   `<brand>can.py`:  Helpers for building CAN messages.
*   `fingerprints.py`: ECU firmware identification.
*   `interface.py`: High-level car interface class.
*   `radar_interface.py`: Parses radar data.
*   `values.py`: Enumeration of supported cars.

### Reverse Engineer CAN Messages

Use [cabana](https://github.com/commaai/openpilot/tree/master/tools/cabana) to analyze CAN data and identify relevant messages by recording driving data.

### Tuning

Use the [longitudinal maneuvers](https://github.com/commaai/openpilot/tree/master/tools/longitudinal_maneuvers) to tune your car's longitudinal control.

## Contributing

Contribute to opendbc! Coordinate development on GitHub and [Discord](https://discord.comma.ai).

### Roadmap

*   `pip install opendbc`
*   100% type and line coverage.
*   Improve car port efficiency.
*   Enhance state of supported cars.
*   Extend support to every car with LKAS + ACC interfaces
*   Automated lateral and longitudinal control/tuning evaluation
*   Auto-tuning for [lateral](https://blog.comma.ai/090release/#torqued-an-auto-tuner-for-lateral-control) and longitudinal control
*   [Automatic Emergency Braking](https://en.wikipedia.org/wiki/Automated_emergency_braking_system)

### Bounties

Earn rewards for contributions!

*   $2000 - [Any car brand / platform port](https://github.com/orgs/commaai/projects/26/views/1?pane=issue&itemId=47913774)
*   $250 - [Any car model port](https://github.com/orgs/commaai/projects/26/views/1?pane=issue&itemId=47913790)
*   $300 - [Reverse Engineering a new Actuation Message](https://github.com/orgs/commaai/projects/26/views/1?pane=issue&itemId=73445563)

Additional high-value bounties are also available at [comma.ai/bounties](comma.ai/bounties).

## Safety Model

The opendbc safety firmware is critical for [openpilot](https://github.com/commaai/openpilot) and [panda](https://github.com/commaai/panda) operation, providing safety and enforcement. The safety logic is carefully implemented and rigorously tested.

### Code Rigor

*   Static code analysis with [cppcheck](https://github.com/danmar/cppcheck/).
*   [MISRA C:2012](https://misra.org.uk/) compliance.
*   Strict compiler flags.
*   Comprehensive unit tests with 100% line coverage.
*   Mutation testing of MISRA coverage.
*   [ruff linter](https://github.com/astral-sh/ruff) and [mypy](https://mypy-lang.org/) for the car interface library.

## FAQ

**Q: How do I use this?**
**A:** Use with a [comma 3X](https://comma.ai/shop/comma-3x).

**Q: Which cars are supported?**
**A:** See the [supported cars list](docs/CARS.md).

**Q: Can I add support for my car?**
**A:** Yes!  See the [porting guide](https://github.com/commaai/opendbc/blob/docs/README.md#how-to-port-a-car).

**Q: Which cars can be supported?**
**A:** Any car with LKAS and ACC. More info [here](https://github.com/commaai/openpilot/blob/master/docs/CARS.md#dont-see-your-car-here).

**Q: How does this work?**
**A:** This hardware interfaces with your car's built-in lane keep and adaptive cruise features.  See [this talk](https://www.youtube.com/watch?v=FL8CxUSfipM).

**Q: Is there a timeline or roadmap for adding car support?**
**A:** Car support is largely community-driven.

### Terms

*   **port**: Integrating a specific car
*   **lateral control**: Steering control
*   **longitudinal control**: Gas/brakes control
*   **fingerprinting**: Automatic car identification
*   **[LKAS](https://en.wikipedia.org/wiki/Lane_departure_warning_system)**: Lane Keeping Assist
*   **[ACC](https://en.wikipedia.org/wiki/Adaptive_cruise_control)**: Adaptive Cruise Control
*   **[harness](https://comma.ai/shop/car-harness)**: Car-specific hardware
*   **[panda](https://github.com/commaai/panda)**: CAN bus hardware
*   **[ECU](https://en.wikipedia.org/wiki/Electronic_control_unit)**: Car's computer
*   **[CAN bus](https://en.wikipedia.org/wiki/CAN_bus)**: Connects car's ECUs
*   **[cabana](https://github.com/commaai/openpilot/tree/master/tools/cabana#readme)**: CAN message reverse engineering tool
*   **[DBC file](https://en.wikipedia.org/wiki/CAN_bus#DBC)**: Defines CAN messages
*   **[openpilot](https://github.com/commaai/openpilot)**: ADAS system
*   **[comma](https://github.com/commaai)**: The company
*   **[comma 3X](https://comma.ai/shop/comma-3x)**: Hardware to run openpilot

### More resources

* [*How Do We Control The Car?*](https://www.youtube.com/watch?v=nNU6ipme878&pp=ygUoY29tbWEgY29uIDIwMjEgaG93IGRvIHdlIGNvbnRyb2wgdGhlIGNhcg%3D%3D)
* [*How to Port a Car*](https://www.youtube.com/watch?v=XxPS5TpTUnI&t=142s&pp=ygUPamFzb24gY29tbWEgY29u)
* [commaCarSegments](https://huggingface.co/datasets/commaai/commaCarSegments)
* [cabana](https://github.com/commaai/openpilot/tree/master/tools/cabana#readme)
* [can_print_changes.py](https://github.com/commaai/openpilot/blob/master/selfdrive/debug/can_print_changes.py)
* [longitudinal maneuvers](https://github.com/commaai/openpilot/tree/master/tools/longitudinal_maneuvers)
* [opendbc data](https://commaai.github.io/opendbc-data/)

## Join Our Team!

Explore career opportunities at [comma.ai/jobs](https://comma.ai/jobs)! We're always looking for talented engineers to contribute to opendbc and [openpilot](https://github.com/commaai/openpilot).