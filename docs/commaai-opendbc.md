<div align="center" style="text-align: center;">

<h1>opendbc: Open Source Car Interface</h1>
<p>
  <b>opendbc empowers you to control and understand your car's systems.</b>
  <br>
  Gain programmatic access to your vehicle's data and control features like steering, gas, and brakes.
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

## **opendbc: Your Gateway to Vehicle Control**

opendbc is a powerful Python API that provides low-level access to your car's internal systems, allowing you to read and write data for advanced driver-assistance systems (ADAS) and more.  This project, central to the openpilot ecosystem, supports a wide range of vehicles and is continuously expanding its capabilities.  [Explore the original repository](https://github.com/commaai/opendbc).

**Key Features:**

*   **Control & Read Vehicle Systems:**  Directly control and monitor critical functions like steering, acceleration, braking, speed, and steering angle.
*   **Broad Car Compatibility:** Designed to support a wide range of vehicles equipped with LKAS and ACC systems.
*   **DBC File Integration:** Utilizes DBC (Database CAN) files to decode and interpret CAN bus messages, offering a clear understanding of vehicle data.
*   **Open Source & Community Driven:** Benefit from a collaborative environment with active development and community contributions.
*   **Extendable & Customizable:**  Easily add support for new cars and features.
*   **Safety Focused:** Rigorous safety models ensure responsible use and prevent dangerous operation.

## **Getting Started**

### Installation

```bash
git clone https://github.com/commaai/opendbc.git
cd opendbc

# Comprehensive install for dependencies, building, linting, and testing:
./test.sh

# Individual Commands (for greater control):
pip3 install -e .[testing,docs]  # Install dependencies
scons -j8                        # Build with 8 cores
pytest .                         # Run the tests
lefthook run lint                # Run the linter
```

### Example Usage

The [`examples/`](examples/) directory contains practical examples to get you started.  For example, [`examples/joystick.py`](examples/joystick.py) allows you to control your car using a joystick.

## **Project Structure**

*   [`opendbc/dbc/`](opendbc/dbc/): Repository of DBC files used for decoding CAN messages.
*   [`opendbc/can/`](opendbc/can/): Library for parsing and constructing CAN messages from DBC files.
*   [`opendbc/car/`](opendbc/car/): High-level Python library for interacting with supported cars.
*   [`opendbc/safety/`](opendbc/safety/): Safety-critical module responsible for enforcing safe operation and preventing unintended actions.

## **Contributing & Car Porting**

### How to Port a Car

The opendbc project relies heavily on community contributions to support new car models. The guide outlines the process of adding support for new car brands and models.

#### **Connect to the Car**

Connect to the car using a [comma 3X](https://comma.ai/shop/comma-3x) and a car harness.
If no harness is available, start with a "developer harness" from comma.ai/shop and crimp on whatever connector you need.

#### **Structure of a port**

A car port lives in `opendbc/car/<brand>/`:
*   `carstate.py`: parses out the relevant information from the CAN stream using the car's DBC file
*   `carcontroller.py`: outputs CAN messages to control the car
*   `<brand>can.py`: thin Python helpers around the DBC file to build CAN messages
*   `fingerprints.py`: database of ECU firmware versions for identifying car models
*   `interface.py`: high level class for interfacing with the car
*   `radar_interface.py`: parses out the radar
*   `values.py`: enumerates the brand's supported cars

#### **Reverse Engineer CAN messages**

Use [cabana](https://github.com/commaai/openpilot/tree/master/tools/cabana) to reverse engineer CAN messages and interpret their function.

#### **Tuning**

Use the [longitudinal maneuvers](https://github.com/commaai/openpilot/tree/master/tools/longitudinal_maneuvers) report to evaluate your car's longitudinal control and tune it.

### Contributing Guidelines

All development is coordinated through GitHub and [Discord](https://discord.comma.ai). Engage in the `#dev-opendbc-cars` channel.

### Roadmap

*   [ ] `pip install opendbc`
*   [ ] 100% type coverage
*   [ ] 100% line coverage
*   [ ] Make car ports easier: refactors, tools, tests, and docs
*   [ ] Expose the state of all supported cars better: https://github.com/commaai/opendbc/issues/1144
*   [ ] Extend support to every car with LKAS + ACC interfaces
*   [ ] Automatic lateral and longitudinal control/tuning evaluation
*   [ ] Auto-tuning for [lateral](https://blog.comma.ai/090release/#torqued-an-auto-tuner-for-lateral-control) and longitudinal control
*   [ ] [Automatic Emergency Braking](https://en.wikipedia.org/wiki/Automated_emergency_braking_system)

Contributions are welcome!

## **Safety Model**

The `opendbc/safety` module implements a safety model designed to ensure safe operation. This module, used in conjunction with [openpilot](https://github.com/commaai/openpilot) and the [panda](https://comma.ai/shop/panda), enforces safety protocols and prevents unintended vehicle behavior. The safety firmware is written for its use in conjunction with [openpilot](https://github.com/commaai/openpilot) and [panda](https://comma.ai/panda). The safety firmware, through its safety model, provides and enforces the [openpilot safety](https://github.com/commaai/openpilot/blob/master/docs/SAFETY.md).

### Code Rigor

The code undergoes rigorous testing to ensure safety:

*   Static code analysis using [cppcheck](https://github.com/danmar/cppcheck/), including MISRA C:2012 compliance checks.
*   Strict compiler flags (`-Wall -Wextra -Wstrict-prototypes -Werror`).
*   Extensive unit tests covering the safety logic ([opendbc/safety/tests](opendbc/safety/tests)).
*   Mutation tests on MISRA coverage.
*   100% line coverage enforced for unit tests.
*   Ruff linter and mypy for the car interface library.

## **Bounties**

We offer bounties for contributions:

*   $2000 - [Any car brand / platform port](https://github.com/orgs/commaai/projects/26/views/1?pane=issue&itemId=47913774)
*   $250 - [Any car model port](https://github.com/orgs/commaai/projects/26/views/1?pane=issue&itemId=47913790)
*   $300 - [Reverse Engineering a new Actuation Message](https://github.com/orgs/commaai/projects/26/views/1?pane=issue&itemId=73445563)

Additional bounties are available for more popular car models. See [comma.ai/bounties](comma.ai/bounties).

## **Frequently Asked Questions (FAQ)**

*   **How do I use this?** The [comma 3X](https://comma.ai/shop/comma-3x) is recommended.
*   **Which cars are supported?** See the [supported cars list](docs/CARS.md).
*   **Can I add support for my car?** Yes. Refer to the car porting guide [here](https://github.com/commaai/opendbc/blob/docs/README.md#how-to-port-a-car).
*   **Which cars can be supported?** Any car with LKAS and ACC. More info [here](https://github.com/commaai/openpilot/blob/master/docs/CARS.md#dont-see-your-car-here).
*   **How does this work?** Replaces your car's built-in lane keep and adaptive cruise features. See [this talk](https://www.youtube.com/watch?v=FL8CxUSfipM).
*   **Is there a timeline or roadmap for adding car support?** Community-driven, with comma validating the final product.

### **Key Terms**

*   **port**: Adding integration and support for a specific car
*   **lateral control**: Steering control
*   **longitudinal control**: Gas/brakes control
*   **fingerprinting**: Identifying the car
*   **[LKAS](https://en.wikipedia.org/wiki/Lane_departure_warning_system)**: Lane keeping assist
*   **[ACC](https://en.wikipedia.org/wiki/Adaptive_cruise_control)**: Adaptive cruise control
*   **[harness](https://comma.ai/shop/car-harness)**: Hardware to intercept ADAS messages
*   **[panda](https://github.com/commaai/panda)**: Hardware for CAN bus access
*   **[ECU](https://en.wikipedia.org/wiki/Electronic_control_unit)**: Car control modules
*   **[CAN bus](https://en.wikipedia.org/wiki/CAN_bus)**: Car communication bus
*   **[cabana](https://github.com/commaai/openpilot/tree/master/tools/cabana#readme)**: Tool to reverse engineer CAN messages
*   **[DBC file](https://en.wikipedia.org/wiki/CAN_bus#DBC)**: Definitions for CAN messages
*   **[openpilot](https://github.com/commaai/openpilot)**: An ADAS system for cars supported by opendbc
*   **[comma](https://github.com/commaai)**: The company behind opendbc
*   **[comma 3X](https://comma.ai/shop/comma-3x)**: The hardware to run openpilot

### **More Resources**

*   [*How Do We Control The Car?*](https://www.youtube.com/watch?v=nNU6ipme878&pp=ygUoY29tbWEgY29uIDIwMjEgaG93IGRvIHdlIGNvbnRyb2wgdGhlIGNhcg%3D%3D) by [@robbederks](https://github.com/robbederks) from COMMA_CON 2021
*   [*How to Port a Car*](https://www.youtube.com/watch?v=XxPS5TpTUnI&t=142s&pp=ygUPamFzb24gY29tbWEgY29u) by [@jyoung8607](https://github.com/jyoung8607) from COMMA_CON 2023
*   [commaCarSegments](https://huggingface.co/datasets/commaai/commaCarSegments): CAN data from 300 different car models
*   [cabana](https://github.com/commaai/openpilot/tree/master/tools/cabana#readme): reverse engineering CAN messages tool
*   [can_print_changes.py](https://github.com/commaai/openpilot/blob/master/selfdrive/debug/can_print_changes.py): Diff the whole CAN bus across two drives
*   [longitudinal maneuvers](https://github.com/commaai/openpilot/tree/master/tools/longitudinal_maneuvers): Tool for evaluating longitudinal control
*   [opendbc data](https://commaai.github.io/opendbc-data/): repository of longitudinal maneuver evaluations

## **Join the Team - [comma.ai/jobs](https://comma.ai/jobs)**

comma is hiring engineers to work on opendbc and [openpilot](https://github.com/commaai/openpilot). We welcome contributors.