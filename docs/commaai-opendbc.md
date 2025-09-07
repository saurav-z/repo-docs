<div align="center" style="text-align: center;">

<h1>opendbc: Python API for Your Car</h1>
<p>
  <b>Take control of your vehicle's systems with opendbc, the open-source Python API.</b>
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

##  opendbc: Unlock Your Car's Potential 

opendbc is a powerful Python API designed to interface with your car's internal systems. Built by comma.ai, it allows you to read and control various vehicle functions, including steering, gas, brakes, and more.  This project provides the foundation for advanced vehicle control and is a key component of the [openpilot](https://github.com/commaai/openpilot) project. 

**Key Features:**

*   **Control & Read:** Access and manipulate steering, gas, brakes, and other essential car functions.
*   **Broad Compatibility:** Designed to support a wide range of vehicles equipped with features like LKAS and ACC.
*   **Open-Source:**  Benefit from community contributions and the ability to customize and extend the functionality.
*   **Active Development:** Continuously updated and improved to support new car models and features.
*   **Extensive Documentation:** Comprehensive guides and resources for usage, contribution, and car porting.
*   **Safety Focused:** Rigorous code rigor and safety model to ensure safe operation.

[**Explore the opendbc Repository on GitHub**](https://github.com/commaai/opendbc)

---

## Getting Started

### Installation

```bash
git clone https://github.com/commaai/opendbc.git
cd opendbc

# Install dependencies, build, lint, and run tests (recommended)
./test.sh

# Individual commands for manual installation
pip3 install -e .[testing,docs]  # install dependencies
scons -j8                        # build with 8 cores
pytest .                         # run the tests
lefthook run lint                # run the linter
```

### Examples

The `examples/` directory provides practical programs to get you started:

*   [`examples/joystick.py`](examples/joystick.py): Control your car with a joystick.

### Project Structure

*   `opendbc/dbc/`: Contains [DBC](https://en.wikipedia.org/wiki/CAN_bus#DBC_(CAN_Database_Files)) files for various car models.
*   `opendbc/can/`: A library for parsing and constructing CAN messages using DBC files.
*   `opendbc/car/`:  A high-level library for interacting with cars using Python.
*   `opendbc/safety/`: Ensures functional safety for all supported vehicles in `opendbc/car/`.

## How to Port a Car

Extend opendbc by adding support for new car models. The process involves:

1.  **Connect to the Car:** Use a [comma 3X](https://comma.ai/shop/comma-3x) and a compatible car harness. (or build your own)
2.  **Understand the Port Structure:** The port consists of `carstate.py`, `carcontroller.py`, `<brand>can.py`, `fingerprints.py`, `interface.py`, `radar_interface.py`, and `values.py`.
3.  **Reverse Engineer CAN Messages:** Use [cabana](https://github.com/commaai/openpilot/tree/master/tools/cabana) to analyze CAN data and identify relevant messages.
4.  **Tuning:** Utilize tools like the [longitudinal maneuvers](https://github.com/commaai/openpilot/tree/master/tools/longitudinal_maneuvers) report to optimize control.

## Contributing

Contribute to opendbc!  All development is coordinated on GitHub and [Discord](https://discord.comma.ai). Check out the `#dev-opendbc-cars` channel and the `Vehicle Specific` section.

### Roadmap

*   `pip install opendbc`
*   100% type coverage
*   100% line coverage
*   Make car ports easier: refactors, tools, tests, and docs
*   Expose the state of all supported cars better: https://github.com/commaai/opendbc/issues/1144
*   Extend support to every car with LKAS + ACC interfaces
*   Automatic lateral and longitudinal control/tuning evaluation
*   Auto-tuning for [lateral](https://blog.comma.ai/090release/#torqued-an-auto-tuner-for-lateral-control) and longitudinal control
*   [Automatic Emergency Braking](https://en.wikipedia.org/wiki/Automated_emergency_braking_system)

### Bounties

Contribute to opendbc and get paid!

*   $2000 - [Any car brand / platform port](https://github.com/orgs/commaai/projects/26/views/1?pane=issue&itemId=47913774)
*   $250 - [Any car model port](https://github.com/orgs/commaai/projects/26/views/1?pane=issue&itemId=47913790)
*   $300 - [Reverse Engineering a new Actuation Message](https://github.com/orgs/commaai/projects/26/views/1?pane=issue&itemId=73445563)

Find higher value bounties for popular cars at [comma.ai/bounties](comma.ai/bounties).

## Safety Model

The opendbc safety firmware is engineered to work with [openpilot](https://github.com/commaai/openpilot) and [panda](https://github.com/commaai/panda) to maintain the [openpilot safety](https://github.com/commaai/openpilot/blob/master/docs/SAFETY.md) model.

Safety modes optionally support `controls_allowed`, which allows or blocks a subset of messages based on a customizable state in the board.

## Code Rigor

Rigorous testing and code analysis are performed to ensure the reliability and safety of the `opendbc/safety` firmware:

*   Static code analysis using [cppcheck](https://github.com/danmar/cppcheck/), including [MISRA C:2012](https://misra.org.uk/) compliance.
*   Strict compiler flags: `-Wall -Wextra -Wstrict-prototypes -Werror`.
*   Unit tests and [mutation tests](opendbc/safety/tests/misra/test_mutation.py) for thorough verification.
*   100% line coverage on safety unit tests.
*   Ruff linter and [mypy](https://mypy-lang.org/) for the car interface library.

## FAQ

*   **How do I use this?** A [comma 3X](https://comma.ai/shop/comma-3x) is designed to run opendbc and openpilot.
*   **Which cars are supported?** See the [supported cars list](docs/CARS.md).
*   **Can I add support for my car?** Yes!  See the car porting guide [here](https://github.com/commaai/opendbc/blob/docs/README.md#how-to-port-a-car).
*   **Which cars can be supported?** Any car with LKAS and ACC. More info [here](https://github.com/commaai/openpilot/blob/master/docs/CARS.md#dont-see-your-car-here).
*   **How does this work?** Hardware is designed to replace the car's built-in lane keep and adaptive cruise features. See [this talk](https://www.youtube.com/watch?v=FL8CxUSfipM) for an in-depth explanation.
*   **Is there a timeline or roadmap for adding car support?** No, community contributions are key, with comma doing final safety validation.

### Terms

*   **port**: Integration and support of a specific car
*   **lateral control**: Steering control
*   **longitudinal control**: Gas/brakes control
*   **fingerprinting**: Automatic car identification
*   **[LKAS](https://en.wikipedia.org/wiki/Lane_departure_warning_system)**: Lane keeping assist
*   **[ACC](https://en.wikipedia.org/wiki/Adaptive_cruise_control)**: Adaptive cruise control
*   **[harness](https://comma.ai/shop/car-harness)**: Car-specific hardware for ADAS message interception
*   **[panda](https://github.com/commaai/panda)**: Hardware for accessing a car's CAN bus
*   **[ECU](https://en.wikipedia.org/wiki/Electronic_control_unit)**: In-car control modules
*   **[CAN bus](https://en.wikipedia.org/wiki/CAN_bus)**: Car's internal communication network
*   **[cabana](https://github.com/commaai/openpilot/tree/master/tools/cabana#readme)**: Tool for reverse engineering CAN messages
*   **[DBC file](https://en.wikipedia.org/wiki/CAN_bus#DBC)**: Message definitions for the CAN bus
*   **[openpilot](https://github.com/commaai/openpilot)**: ADAS system using opendbc
*   **[comma](https://github.com/commaai)**: The company behind opendbc
*   **[comma 3X](https://comma.ai/shop/comma-3x)**: Hardware for running openpilot

### More Resources

*   [*How Do We Control The Car?*](https://www.youtube.com/watch?v=nNU6ipme878&pp=ygUoY29tbWEgY29uIDIwMjEgaG93IGRvIHdlIGNvbnRyb2wgdGhlIGNhcg%3D%3D) by [@robbederks](https://github.com/robbederks) from COMMA_CON 2021
*   [*How to Port a Car*](https://www.youtube.com/watch?v=XxPS5TpTUnI&t=142s&pp=ygUPamFzb24gY29tbWEgY29u) by [@jyoung8607](https://github.com/jyoung8607) from COMMA_CON 2023
*   [commaCarSegments](https://huggingface.co/datasets/commaai/commaCarSegments): A massive CAN data dataset
*   [cabana](https://github.com/commaai/openpilot/tree/master/tools/cabana#readme): A tool for reverse engineering CAN messages
*   [can_print_changes.py](https://github.com/commaai/openpilot/blob/master/selfdrive/debug/can_print_changes.py): Diff the whole CAN bus across two drives
*   [longitudinal maneuvers](https://github.com/commaai/openpilot/tree/master/tools/longitudinal_maneuvers): A tool for evaluating and tuning longitudinal control
*   [opendbc data](https://commaai.github.io/opendbc-data/): Longitudinal maneuver evaluations

## Join the Team - [comma.ai/jobs](https://comma.ai/jobs)

comma is looking for engineers to work on opendbc and [openpilot](https://github.com/commaai/openpilot), and welcomes contributions from the community!