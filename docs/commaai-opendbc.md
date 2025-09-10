<div align="center" style="text-align: center;">

<h1>opendbc: Open-Source Car Control and Data Acquisition</h1>
<p>
  <b>Unlock your car's potential with opendbc, a Python API for advanced vehicle control and data analysis.</b>
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

opendbc empowers you to control and monitor your car's systems, offering a powerful interface for developers and enthusiasts.  Leverage the power of open-source to modify gas, brakes, steering, and access vital vehicle data. This project focuses on supporting advanced driver-assistance systems (ADAS) interfaces for openpilot, offering a gateway to building the best vehicle management applications.

**Key Features:**

*   **Control:** Send commands to control steering, gas, and brakes.
*   **Data Acquisition:** Read real-time data like speed and steering angle.
*   **Open Source:** Contribute to a community-driven project.
*   **Extensible:**  Support for a growing list of vehicles and features.
*   **Safety Focused:** Rigorous code rigor and safety model.

**[Explore the opendbc Repository on GitHub](https://github.com/commaai/opendbc)**

---

opendbc supports a wide array of vehicles, with new support being continuously added by the community.

## Quick Start

```bash
git clone https://github.com/commaai/opendbc.git
cd opendbc

# Install dependencies, build, lint, and test (all-in-one)
./test.sh

# Individual commands:
pip3 install -e .[testing,docs]  # install dependencies
scons -j8                        # build with 8 cores
pytest .                         # run the tests
lefthook run lint                # run the linter
```

Example programs in the [`examples/`](examples/) directory read car state and control steering, gas, and brakes.  Control your car with a joystick using [`examples/joystick.py`](examples/joystick.py).

## Project Structure

*   [`opendbc/dbc/`](opendbc/dbc/):  DBC (CAN Database Files) repository.
*   [`opendbc/can/`](opendbc/can/): CAN message parsing and building library.
*   [`opendbc/car/`](opendbc/car/): High-level Python interface for car interaction.
*   [`opendbc/safety/`](opendbc/safety/): Functional safety module for supported vehicles.

## How to Port a Car

Learn how to add support for your car, including everything from adding basic steering control to advanced features.

### Connect to the Car

Connect to your car using a comma 3X and a car harness. Check comma.ai/shop for compatible harnesses.

### Structure of a Port

A car port typically includes the following components, located within `opendbc/car/<brand>/`:
*   `carstate.py`: Parses CAN data.
*   `carcontroller.py`: Sends CAN control messages.
*   `<brand>can.py`: Helper functions for building CAN messages.
*   `fingerprints.py`: ECU firmware version database.
*   `interface.py`: High-level car interface class.
*   `radar_interface.py`: Radar data parsing (if applicable).
*   `values.py`: Enumerates supported car models.

### Reverse Engineer CAN Messages

Analyze car data using [cabana](https://github.com/commaai/openpilot/tree/master/tools/cabana) to understand CAN message structures.

### Tuning

Use the [longitudinal maneuvers](https://github.com/commaai/openpilot/tree/master/tools/longitudinal_maneuvers) report to tune your car's longitudinal control.

## Contributing

Contribute to opendbc on GitHub and Discord.  Join the `#dev-opendbc-cars` channel and explore the `Vehicle Specific` section.

### Roadmap

*   [ ] `pip install opendbc`
*   [ ] 100% type coverage
*   [ ] 100% line coverage
*   [ ] Improve car port creation through refactoring and tooling.
*   [ ] Expand support to every car with LKAS + ACC.
*   [ ] Auto-tuning for lateral and longitudinal control.
*   [ ] Automatic Emergency Braking (AEB).

## Safety Model

The opendbc safety firmware, used with [openpilot](https://github.com/commaai/openpilot) and [panda](https://github.com/commaai/panda), provides and enforces [openpilot safety](https://github.com/commaai/openpilot/blob/master/docs/SAFETY.md).
By default, [panda's](https://comma.ai/shop/panda) safety is silent. Safety modes optionally support `controls_allowed` for customizable message control based on board state.

## Code Rigor

The safety firmware code within the `safety` folder adheres to strict standards:

*   Static code analysis with [cppcheck](https://github.com/danmar/cppcheck/) and [MISRA C:2012](https://misra.org.uk/) checks.
*   Strict compiler flags (`-Wall -Wextra -Wstrict-prototypes -Werror`).
*   Unit tests for safety logic with 100% line coverage.
*   Mutation testing for MISRA coverage.
*   Ruff linter and mypy checks for car interface library.

### Bounties

Earn bounties for your contributions:

*   $2000 - [Any car brand / platform port](https://github.com/orgs/commaai/projects/26/views/1?pane=issue&itemId=47913774)
*   $250 - [Any car model port](https://github.com/orgs/commaai/projects/26/views/1?pane=issue&itemId=47913790)
*   $300 - [Reverse Engineering a new Actuation Message](https://github.com/orgs/commaai/projects/26/views/1?pane=issue&itemId=73445563)

More bounties available at [comma.ai/bounties](comma.ai/bounties).

## FAQ

*   **How do I use this?** Use it with a [comma 3X](https://comma.ai/shop/comma-3x).
*   **Which cars are supported?** Check the [supported cars list](docs/CARS.md).
*   **Can I add support for my car?**  Yes!  See the guide [here](https://github.com/commaai/opendbc/blob/docs/README.md#how-to-port-a-car).
*   **Which cars can be supported?**  Cars with LKAS and ACC. More info [here](https://github.com/commaai/openpilot/blob/master/docs/CARS.md#dont-see-your-car-here).
*   **How does this work?** Replace your car's lane keep and adaptive cruise features. See [this talk](https://www.youtube.com/watch?v=FL8CxUSfipM).
*   **Is there a timeline/roadmap?** Community-driven car support with comma's validation.

### Terms

*   **port**: Integrate and support a car
*   **lateral control**: Steering control
*   **longitudinal control**: Gas/brakes control
*   **fingerprinting**: Auto car identification
*   **[LKAS](https://en.wikipedia.org/wiki/Lane_departure_warning_system)**: Lane Keeping Assist
*   **[ACC](https://en.wikipedia.org/wiki/Adaptive_cruise_control)**: Adaptive Cruise Control
*   **[harness](https://comma.ai/shop/car-harness)**: Hardware to connect to the car.
*   **[panda](https://github.com/commaai/panda)**: Hardware for CAN bus access.
*   **[ECU](https://en.wikipedia.org/wiki/Electronic_control_unit)**: Car's computers/control modules
*   **[CAN bus](https://en.wikipedia.org/wiki/CAN_bus)**: Communication bus in cars
*   **[cabana](https://github.com/commaai/openpilot/tree/master/tools/cabana#readme)**: Tool for reverse engineering CAN messages
*   **[DBC file](https://en.wikipedia.org/wiki/CAN_bus#DBC)**: Defines messages on a CAN bus
*   **[openpilot](https://github.com/commaai/openpilot)**: ADAS system using opendbc
*   **[comma](https://github.com/commaai)**: The company behind opendbc
*   **[comma 3X](https://comma.ai/shop/comma-3x)**: Hardware to run openpilot

### More Resources

*   [*How Do We Control The Car?*](https://www.youtube.com/watch?v=nNU6ipme878&pp=ygUoY29tbWEgY29uIDIwMjEgaG93IGRvIHdlIGNvbnRyb2wgdGhlIGNhcg%3D%3D) by [@robbederks](https://github.com/robbederks) from COMMA_CON 2021
*   [*How to Port a Car*](https://www.youtube.com/watch?v=XxPS5TpTUnI&t=142s&pp=ygUPamFzb24gY29tbWEgY29u) by [@jyoung8607](https://github.com/jyoung8607) from COMMA_CON 2023
*   [commaCarSegments](https://huggingface.co/datasets/commaai/commaCarSegments): Dataset of CAN data.
*   [cabana](https://github.com/commaai/openpilot/tree/master/tools/cabana#readme): Tool for reverse engineering CAN messages
*   [can_print_changes.py](https://github.com/commaai/openpilot/blob/master/selfdrive/debug/can_print_changes.py): Diff the whole CAN bus across two drives
*   [longitudinal maneuvers](https://github.com/commaai/openpilot/tree/master/tools/longitudinal_maneuvers): Tool for evaluating and tuning longitudinal control
*   [opendbc data](https://commaai.github.io/opendbc-data/): Repository of longitudinal maneuver evaluations

## Join the Team – [comma.ai/jobs](https://comma.ai/jobs)

comma is hiring engineers.  We welcome contributions!