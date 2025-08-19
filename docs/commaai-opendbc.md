<div align="center" style="text-align: center;">

<h1>opendbc: Open Car Database and Control</h1>
<p>
  <b>Take control of your car's systems with opendbc, a powerful Python API.</b>
  <br>
  Unlock advanced control over steering, gas, brakes, and more, while gaining real-time access to crucial vehicle data.
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

opendbc empowers you to interact with your car's internal systems, enabling a wide range of applications from advanced driver-assistance systems (ADAS) to custom vehicle management solutions.  Built for developers and automotive enthusiasts, opendbc provides the tools to read and control your car's data.  This project aims to support controlling the steering, gas, and brakes on every single car that has LKAS and ACC interfaces.  

**[Explore the opendbc repository on GitHub](https://github.com/commaai/opendbc)**

**Key Features:**

*   **Precise Control:** Manipulate steering, acceleration, and braking with ease.
*   **Real-time Data Access:** Read vehicle speed, steering angle, and other vital information.
*   **Extensive Car Support:**  Supports a wide range of vehicles with expanding compatibility.
*   **Flexible API:**  Python-based API for easy integration and customization.
*   **Community-Driven:** Benefit from a thriving community and open-source collaboration.
*   **Safety Focused:** Integrated safety features for responsible vehicle control.

---

## Getting Started

Quickly set up opendbc for development:

```bash
git clone https://github.com/commaai/opendbc.git
cd opendbc

# Run an all-in-one dependency installation, compilation, linting, and tests
./test.sh

# Or, run the individual commands:
pip3 install -e .[testing,docs]  # Install dependencies
scons -j8                        # Build with 8 cores
pytest .                         # Run the tests
lefthook run lint                # Run the linter
```

Explore example programs to read car data and control vehicle functions: `examples/joystick.py` lets you control a car with a joystick.

## Project Structure

*   `opendbc/dbc/`: Stores [DBC](https://en.wikipedia.org/wiki/CAN_bus#DBC) files for car communication.
*   `opendbc/can/`: Provides a library for parsing and building CAN messages from DBC files.
*   `opendbc/car/`: Offers a high-level Python library for interacting with cars.
*   `opendbc/safety/`: Implements safety features for supported vehicles.

## Contributing

Contribute to the open-source project! Development is coordinated on [GitHub](https://github.com/commaai/opendbc) and [Discord](https://discord.comma.ai). Check out the `#dev-opendbc-cars` channel and `Vehicle Specific` section.

### Roadmap

*   \[ ] `pip install opendbc`
*   \[ ] 100% type coverage
*   \[ ] 100% line coverage
*   \[ ] Make car ports easier: refactors, tools, tests, and docs
*   \[ ] Expose the state of all supported cars better: https://github.com/commaai/opendbc/issues/1144

### Roadmap (Longer Term)

*   \[ ] Extend support to every car with LKAS + ACC interfaces
*   \[ ] Automatic lateral and longitudinal control/tuning evaluation
*   \[ ] Auto-tuning for [lateral](https://blog.comma.ai/090release/#torqued-an-auto-tuner-for-lateral-control) and longitudinal control
*   \[ ] [Automatic Emergency Braking](https://en.wikipedia.org/wiki/Automated_emergency_braking_system)

## How to Port a Car

Add support for new car models:

1.  **Connect to the Car:** Use a comma 3X and a car harness (available at comma.ai/shop).
2.  **Reverse Engineer CAN messages:** Record and analyze data using [cabana](https://github.com/commaai/openpilot/tree/master/tools/cabana).
3.  **Structure of a port:** Each port lives in `opendbc/car/<brand>/`:
    *   `carstate.py`: Parses data from the CAN stream.
    *   `carcontroller.py`: Outputs CAN messages to control the car.
    *   `<brand>can.py`: Builds CAN messages using the car's DBC file.
    *   `fingerprints.py`:  Identifies car models.
    *   `interface.py`: High-level car interface class.
    *   `radar_interface.py`: Parses radar data.
    *   `values.py`: Defines supported cars.
4.  **Tuning:** Use the [longitudinal maneuvers](https://github.com/commaai/openpilot/tree/master/tools/longitudinal_maneuvers) report to evaluate and tune car control.

## Safety Model

The opendbc safety firmware, designed for use with [openpilot](https://github.com/commaai/openpilot) and [panda](https://comma.ai/shop/panda), defaults to `SAFETY_SILENT` mode for security.  Select a safety mode to send messages.

## Code Rigor

The safety firmware within the `safety` folder undergoes rigorous testing:

*   Static code analysis with [cppcheck](https://github.com/danmar/cppcheck/).
*   MISRA C:2012 compliance checks with [cppcheck](https://github.com/danmar/cppcheck/) addon (see [current coverage](opendbc/safety/tests/misra/coverage_table)).
*   Strict compiler flags: `-Wall -Wextra -Wstrict-prototypes -Werror`.
*   Unit tests for safety logic in `opendbc/safety/tests`.
*   Mutation testing on MISRA coverage.
*   100% line coverage enforced on safety unit tests.
*   [Ruff linter](https://github.com/astral-sh/ruff) and [mypy](https://mypy-lang.org/) on car interface library.

## Bounties

Contribute and earn!

*   $2000 - [Any car brand / platform port](https://github.com/orgs/commaai/projects/26/views/1?pane=issue&itemId=47913774)
*   $250 - [Any car model port](https://github.com/orgs/commaai/projects/26/views/1?pane=issue&itemId=47913790)
*   $300 - [Reverse Engineering a new Actuation Message](https://github.com/orgs/commaai/projects/26/views/1?pane=issue&itemId=73445563)

Higher value bounties are available for popular car models at [comma.ai/bounties](comma.ai/bounties).

## FAQ

*   **How do I use this?**  A [comma 3X](https://comma.ai/shop/comma-3x) is the recommended hardware.
*   **Which cars are supported?** See the [supported cars list](docs/CARS.md).
*   **Can I add support for my car?** Yes, follow the guide [here](https://github.com/commaai/opendbc/blob/docs/README.md#how-to-port-a-car).
*   **Which cars can be supported?** Any car with LKAS and ACC. More info [here](https://github.com/commaai/openpilot/blob/master/docs/CARS.md#dont-see-your-car-here).
*   **How does this work?**  Hardware replaces your car's built-in features. See [this talk](https://www.youtube.com/watch?v=FL8CxUSfipM) for more details.
*   **Is there a timeline or roadmap for adding car support?** Community-driven, with comma doing final safety validation.

## Terms

*   **port**:  Car integration and support
*   **lateral control**:  Steering control
*   **longitudinal control**:  Gas/brake control
*   **fingerprinting**:  Car identification
*   **[LKAS](https://en.wikipedia.org/wiki/Lane_departure_warning_system)**:  Lane keeping assist
*   **[ACC](https://en.wikipedia.org/wiki/Adaptive_cruise_control)**:  Adaptive cruise control
*   **[harness](https://comma.ai/shop/car-harness)**: Car-specific hardware
*   **[panda](https://github.com/commaai/panda)**: Hardware for CAN bus access
*   **[ECU](https://en.wikipedia.org/wiki/Electronic_control_unit)**:  Car's control modules
*   **[CAN bus](https://en.wikipedia.org/wiki/CAN_bus)**:  Network for ECUs
*   **[cabana](https://github.com/commaai/openpilot/tree/master/tools/cabana#readme)**:  Tool for CAN message analysis
*   **[DBC file](https://en.wikipedia.org/wiki/CAN_bus#DBC)**:  CAN bus message definitions
*   **[openpilot](https://github.com/commaai/openpilot)**:  ADAS system using opendbc
*   **[comma](https://github.com/commaai)**:  The company behind opendbc
*   **[comma 3X](https://comma.ai/shop/comma-3x)**:  The recommended hardware

## More Resources

*   [*How Do We Control The Car?*](https://www.youtube.com/watch?v=nNU6ipme878&pp=ygUoY29tbWEgY29uIDIwMjEgaG93IGRvIHdlIGNvbnRyb2wgdGhlIGNhcg%3D%3D) by [@robbederks](https://github.com/robbederks)
*   [*How to Port a Car*](https://www.youtube.com/watch?v=XxPS5TpTUnI&t=142s&pp=ygUPamFzb24gY29tbWEgY29u) by [@jyoung8607](https://github.com/jyoung8607)
*   [commaCarSegments](https://huggingface.co/datasets/commaai/commaCarSegments): CAN data dataset
*   [cabana](https://github.com/commaai/openpilot/tree/master/tools/cabana#readme): CAN message reverse engineering
*   [can_print_changes.py](https://github.com/commaai/openpilot/blob/master/selfdrive/debug/can_print_changes.py): CAN bus diffing
*   [longitudinal maneuvers](https://github.com/commaai/openpilot/tree/master/tools/longitudinal_maneuvers): Longitudinal control evaluation
*   [opendbc data](https://commaai.github.io/opendbc-data/): Longitudinal maneuver evaluations

## Come work with us -- [comma.ai/jobs](https://comma.ai/jobs)

Join comma's team of engineers! We're hiring contributors to work on opendbc and [openpilot](https://github.com/commaai/openpilot).