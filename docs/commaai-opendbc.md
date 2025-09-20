<div align="center" style="text-align: center;">

<h1>opendbc: Your Python API for Car Control</h1>
<p>
  <b>Take control of your car with opendbc, the open-source API for ADAS systems, enabling you to control gas, brakes, and steering, and read vital vehicle data.</b>
  <br>
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

## Unleash the Power of Your Car with opendbc

opendbc empowers you to interact with your car's systems using a Python API. This project provides the tools and resources to:

*   **Control**: Manipulate steering, gas, and brakes.
*   **Read Data**: Access real-time information like speed and steering angle.
*   **Extend**: Contribute to supporting a wide range of vehicles.

Built to support [openpilot](https://github.com/commaai/openpilot), opendbc is the foundation for advanced vehicle control and management.

**[View the opendbc Repository on GitHub](https://github.com/commaai/opendbc)**

### Key Features

*   **Python-Based API:** Easy-to-use Python API for interacting with your car.
*   **DBC File Integration:** Utilizes DBC files to parse and build CAN messages.
*   **Car-Specific Libraries:** High-level libraries designed for interfacing with different car brands.
*   **Safety-Focused:** Includes a safety model to ensure responsible vehicle control.
*   **Community Driven:** Open for contributions and collaboration on supported vehicles.
*   **Extensive Documentation:** Comprehensive guides, including supported cars lists and contributor guides, for a streamlined integration.
*   **Built-in Tests:** All-in-one test suite, including individual tests.

---

## Quick Start

Get up and running quickly:

```bash
git clone https://github.com/commaai/opendbc.git
cd opendbc

# Install dependencies, build, and run tests:
./test.sh
```

For more detailed instructions and example programs, see the `examples/` directory, including `examples/joystick.py`.

### Project Structure

*   `opendbc/dbc/`: Contains [DBC](https://en.wikipedia.org/wiki/CAN_bus#DBC_(CAN_Database_Files)) files.
*   `opendbc/can/`: Library for parsing and building CAN messages.
*   `opendbc/car/`: High-level library for interacting with cars using Python.
*   `opendbc/safety/`: Safety features for all supported cars.

---

## How to Port a Car

Expand your car's compatibility with opendbc.

### Steps

1.  **Connect to the Car**: Connect to the CAN bus using a comma 3X and a car harness.
2.  **Structure of a Port**: Each car port resides in `opendbc/car/<brand>/` and includes:
    *   `carstate.py`: Parses CAN data.
    *   `carcontroller.py`: Outputs control messages.
    *   `<brand>can.py`: Python helpers for CAN message creation.
    *   `fingerprints.py`: Identifies car models.
    *   `interface.py`: High-level class for interfacing.
    *   `radar_interface.py`: Parses radar data.
    *   `values.py`: Defines supported car models.
3.  **Reverse Engineer CAN Messages**: Analyze CAN data using tools like [cabana](https://github.com/commaai/openpilot/tree/master/tools/cabana).
4.  **Tuning**: Use tools like the [longitudinal maneuvers](https://github.com/commaai/openpilot/tree/master/tools/longitudinal_maneuvers) report to fine-tune vehicle performance.

---

## Contributing

Join the opendbc community on GitHub and [Discord](https://discord.comma.ai) to contribute.

### Roadmap

**Short Term:**

*   `pip install opendbc`
*   Achieve 100% type and line coverage.
*   Improve car port development.
*   Enhance state exposure for supported cars.

**Longer Term:**

*   Expand support for all cars with LKAS and ACC.
*   Automate control and tuning evaluation.
*   Implement auto-tuning features.
*   Add Automatic Emergency Braking support.

### Bounties

Get rewarded for your contributions:

*   $2000 - [Any car brand / platform port](https://github.com/orgs/commaai/projects/26/views/1?pane=issue&itemId=47913774)
*   $250 - [Any car model port](https://github.com/orgs/commaai/projects/26/views/1?pane=issue&itemId=47913790)
*   $300 - [Reverse Engineering a new Actuation Message](https://github.com/orgs/commaai/projects/26/views/1?pane=issue&itemId=73445563)

Check [comma.ai/bounties](comma.ai/bounties) for additional incentives for popular cars.

---

## Safety Model and Code Rigor

The opendbc safety firmware, designed for use with [openpilot](https://github.com/commaai/openpilot) and [panda](https://github.com/commaai/panda), enforces [openpilot safety](https://github.com/commaai/openpilot/blob/master/docs/SAFETY.md). The codebase emphasizes rigor and quality.

### Code Rigor

*   Static analysis with [cppcheck](https://github.com/danmar/cppcheck/) and [MISRA C:2012](https://misra.org.uk/) checks.
*   Strict compiler flags (`-Wall -Wextra -Wstrict-prototypes -Werror`).
*   Unit tests with 100% line coverage.
*   Mutation testing.
*   [ruff linter](https://github.com/astral-sh/ruff) and [mypy](https://mypy-lang.org/) for the car interface library.

---

## FAQ

***How do I use this?*** A [comma 3X](https://comma.ai/shop/comma-3x) is custom-designed to be the best way to run and develop opendbc and openpilot.

***Which cars are supported?*** See the [supported cars list](docs/CARS.md).

***Can I add support for my car?*** Yes, most car support comes from the community. Read the guide [here](https://github.com/commaai/opendbc/blob/docs/README.md#how-to-port-a-car).

***Which cars can be supported?*** Any car with LKAS and ACC. More info [here](https://github.com/commaai/openpilot/blob/master/docs/CARS.md#dont-see-your-car-here).

***How does this work?*** In short, we designed hardware to replace your car's built-in lane keep and adaptive cruise features. See [this talk](https://www.youtube.com/watch?v=FL8CxUSfipM) for an in-depth explanation.

***Is there a timeline or roadmap for adding car support?*** No, most car support comes from the community, with comma doing final safety and quality validation. The more complete the community car port is and the more popular the car is, the more likely we are to pick it up as the next one to validate.

### Terms

*   **port**: refers to the integration and support of a specific car
*   **lateral control**: aka steering control
*   **longitudinal control**: aka gas/brakes control
*   **fingerprinting**: automatic process for identifying the car
*   **[LKAS](https://en.wikipedia.org/wiki/Lane_departure_warning_system)**: lane keeping assist
*   **[ACC](https://en.wikipedia.org/wiki/Adaptive_cruise_control)**: adaptive cruise control
*   **[harness](https://comma.ai/shop/car-harness)**: car-specific hardware to attach to the car and intercept the ADAS messages
*   **[panda](https://github.com/commaai/panda)**: hardware used to get on a car's CAN bus
*   **[ECU](https://en.wikipedia.org/wiki/Electronic_control_unit)**: computers or control modules inside the car
*   **[CAN bus](https://en.wikipedia.org/wiki/CAN_bus)**: a bus that connects the ECUs in a car
*   **[cabana](https://github.com/commaai/openpilot/tree/master/tools/cabana#readme)**: our tool for reverse engineering CAN messages
*   **[DBC file](https://en.wikipedia.org/wiki/CAN_bus#DBC)**: contains definitions for messages on a CAN bus
*   **[openpilot](https://github.com/commaai/openpilot)**: an ADAS system for cars supported by opendbc
*   **[comma](https://github.com/commaai)**: the company behind opendbc
*   **[comma 3X](https://comma.ai/shop/comma-3x)**: the hardware used to run openpilot

---

### More Resources

*   [*How Do We Control The Car?*](https://www.youtube.com/watch?v=nNU6ipme878&pp=ygUoY29tbWEgY29uIDIwMjEgaG93IGRvIHdlIGNvbnRyb2wgdGhlIGNhcg%3D%3D) by [@robbederks](https://github.com/robbederks) from COMMA_CON 2021
*   [*How to Port a Car*](https://www.youtube.com/watch?v=XxPS5TpTUnI&t=142s&pp=ygUPamFzb24gY29tbWEgY29u) by [@jyoung8607](https://github.com/jyoung8607) from COMMA_CON 2023
*   [commaCarSegments](https://huggingface.co/datasets/commaai/commaCarSegments): a massive dataset of CAN data from 300 different car models
*   [cabana](https://github.com/commaai/openpilot/tree/master/tools/cabana#readme): our tool for reverse engineering CAN messages
*   [can_print_changes.py](https://github.com/commaai/openpilot/blob/master/selfdrive/debug/can_print_changes.py): diff the whole CAN bus across two drives, such as one without any LKAS and one with LKAS
*   [longitudinal maneuvers](https://github.com/commaai/openpilot/tree/master/tools/longitudinal_maneuvers): a tool for evaluating and tuning longitudinal control
*   [opendbc data](https://commaai.github.io/opendbc-data/): a repository of longitudinal maneuver evaluations

---

## Join the Team - [comma.ai/jobs](https://comma.ai/jobs)

comma is hiring engineers to work on opendbc and [openpilot](https://github.com/commaai/openpilot). Contribute and accelerate your career.