<div align="center" style="text-align: center;">
  <h1>opendbc: Python API for Your Car</h1>
  <p><b>Unlock your car's potential: Control steering, gas, and brakes with opendbc!</b></p>

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

opendbc is a powerful Python API enabling you to interact with your car's systems.  Whether you're interested in advanced driver-assistance systems (ADAS) or vehicle management, opendbc provides the tools you need.  Explore the open-source project on [GitHub](https://github.com/commaai/opendbc).

**Key Features:**

*   **Control:** Manipulate steering, gas, and brakes.
*   **Read Data:** Access real-time information like speed and steering angle.
*   **Extensible:** Supports a growing list of car models.
*   **Open Source:** Built for community contribution and innovation.
*   **Integration:** Designed for use with [openpilot](https://github.com/commaai/openpilot) and other applications.

---

## Getting Started

opendbc empowers you to read and write data to your car's systems.

### Quick Installation and Setup

```bash
git clone https://github.com/commaai/opendbc.git
cd opendbc

# Install dependencies, build, run tests, and lint (all-in-one)
./test.sh

# Individual commands for more control
pip3 install -e .[testing,docs]  # install dependencies
scons -j8                        # build with 8 cores
pytest .                         # run the tests
lefthook run lint                # run the linter
```

Explore the `examples/` directory for sample programs demonstrating car interaction, including `examples/joystick.py` for joystick control.

## Project Structure

*   `opendbc/dbc/`:  DBC files repository for CAN message definitions
*   `opendbc/can/`:  Library to parse and build CAN messages
*   `opendbc/car/`:  High-level Python library for car interfacing
*   `opendbc/safety/`: Functional safety components for supported cars

## Porting a Car:  Contribute to the Community

The opendbc project thrives on community contributions!

### Connect to the Car

1.  Connect to your car using a comma 3X and a car harness.
2.  If no harness exists for your car, use a "developer harness" from comma.ai/shop and connect to the CAN buses.

### Structure of a Port

Car port is in `opendbc/car/<brand>/`:
*   `carstate.py`: Parses information from the CAN stream
*   `carcontroller.py`: Outputs CAN messages to control the car
*   `<brand>can.py`: Python helpers around the DBC file
*   `fingerprints.py`: ECU firmware versions for identifying car models
*   `interface.py`: High level class for interfacing with the car
*   `radar_interface.py`: Parses out the radar
*   `values.py`: Enumerates the brand's supported cars

### Reverse Engineering

*   Record a route with interesting events like enabling LKAS and ACC.
*   Load the route in [cabana](https://github.com/commaai/openpilot/tree/master/tools/cabana) to analyze the data.

### Tuning

Use the [longitudinal maneuvers](https://github.com/commaai/openpilot/tree/master/tools/longitudinal_maneuvers) report to evaluate your car's longitudinal control and tune it.

## Contributing

All opendbc development is coordinated on GitHub and [Discord](https://discord.comma.ai). Check out the `#dev-opendbc-cars` channel and `Vehicle Specific` section.

### Roadmap

*   [ ] `pip install opendbc`
*   [ ] 100% type coverage
*   [ ] 100% line coverage
*   [ ] Make car ports easier: refactors, tools, tests, and docs
*   [ ] Expose the state of all supported cars better: https://github.com/commaai/opendbc/issues/1144

## Safety Model: Code Rigor and Confidence

The safety firmware is written in conjunction with [openpilot](https://github.com/commaai/openpilot) and [panda](https://github.com/commaai/panda).

**CI Regression Tests:**

*   Static code analysis with [cppcheck](https://github.com/danmar/cppcheck/).
*   [MISRA C:2012](https://misra.org.uk/) violations checks.
*   Strict compiler options.
*   Unit tests.
*   Mutation tests.
*   Line coverage.
*   Ruff linter and mypy for the car interface library.

## Bounties: Get Rewarded for Your Contributions

*   $2000 - [Any car brand / platform port](https://github.com/orgs/commaai/projects/26/views/1?pane=issue&itemId=47913774)
*   $250 - [Any car model port](https://github.com/orgs/commaai/projects/26/views/1?pane=issue&itemId=47913790)
*   $300 - [Reverse Engineering a new Actuation Message](https://github.com/orgs/commaai/projects/26/views/1?pane=issue&itemId=73445563)

Additional bounties for popular car models are available at [comma.ai/bounties](comma.ai/bounties).

## FAQ

*   **How do I use this?** A [comma 3X](https://comma.ai/shop/comma-3x) is recommended.
*   **Which cars are supported?** See the [supported cars list](docs/CARS.md).
*   **Can I add support for my car?** Yes, see the car port guide [here](https://github.com/commaai/opendbc/blob/docs/README.md#how-to-port-a-car).
*   **Which cars can be supported?** Any car with LKAS and ACC.  More info [here](https://github.com/commaai/openpilot/blob/master/docs/CARS.md#dont-see-your-car-here).
*   **How does this work?** Hardware replaces your car's built-in features. See [this talk](https://www.youtube.com/watch?v=FL8CxUSfipM).
*   **Is there a timeline for adding car support?** No fixed timeline, community contributions drive support.

### Terms

*   **port**: integration and support of a specific car
*   **lateral control**: steering control
*   **longitudinal control**: gas/brakes control
*   **fingerprinting**: automatic car identification
*   **[LKAS](https://en.wikipedia.org/wiki/Lane_departure_warning_system)**: lane keeping assist
*   **[ACC](https://en.wikipedia.org/wiki/Adaptive_cruise_control)**: adaptive cruise control
*   **[harness](https://comma.ai/shop/car-harness)**: car-specific hardware
*   **[panda](https://github.com/commaai/panda)**: hardware for CAN bus access
*   **[ECU](https://en.wikipedia.org/wiki/Electronic_control_unit)**: car computers
*   **[CAN bus](https://en.wikipedia.org/wiki/CAN_bus)**: car communication network
*   **[cabana](https://github.com/commaai/openpilot/tree/master/tools/cabana#readme)**: CAN message analysis tool
*   **[DBC file](https://en.wikipedia.org/wiki/CAN_bus#DBC)**: CAN message definitions
*   **[openpilot](https://github.com/commaai/openpilot)**: ADAS system
*   **[comma](https://github.com/commaai)**: the company behind opendbc
*   **[comma 3X](https://comma.ai/shop/comma-3x)**: hardware for running openpilot

### More Resources

*   [*How Do We Control The Car?*](https://www.youtube.com/watch?v=nNU6ipme878&pp=ygUoY29tbWEgY29uIDIwMjEgaG93IGRvIHdlIGNvbnRyb2wgdGhlIGNhcg%3D%3D) by [@robbederks](https://github.com/robbederks) from COMMA_CON 2021
*   [*How to Port a Car*](https://www.youtube.com/watch?v=XxPS5TpTUnI&t=142s&pp=ygUPamFzb24gY29tbWEgY29u) by [@jyoung8607](https://github.com/jyoung8607) from COMMA_CON 2023
*   [commaCarSegments](https://huggingface.co/datasets/commaai/commaCarSegments): a massive dataset of CAN data from 300 different car models
*   [cabana](https://github.com/commaai/openpilot/tree/master/tools/cabana#readme): our tool for reverse engineering CAN messages
*   [can_print_changes.py](https://github.com/commaai/openpilot/blob/master/selfdrive/debug/can_print_changes.py): diff the whole CAN bus across two drives, such as one without any LKAS and one with LKAS
*   [longitudinal maneuvers](https://github.com/commaai/openpilot/tree/master/tools/longitudinal_maneuvers): a tool for evaluating and tuning longitudinal control
*   [opendbc data](https://commaai.github.io/opendbc-data/): a repository of longitudinal maneuver evaluations

## Join the Team:  [comma.ai/jobs](https://comma.ai/jobs)

comma is hiring engineers to work on opendbc and [openpilot](https://github.com/commaai/openpilot). We love hiring contributors.