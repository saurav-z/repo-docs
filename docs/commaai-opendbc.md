<div align="center" style="text-align: center;">

<h1>opendbc: Your Python API for Automotive Control</h1>

<p>
  <b>Take control of your car!</b>  opendbc empowers you to interface with your vehicle's systems, including steering, gas, brakes, and more.
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

opendbc is a Python API developed by [comma.ai](https://github.com/commaai/opendbc) for controlling and monitoring modern vehicles.  Leveraging the power of CAN bus communication, it allows you to read and write data to your car's systems.

**Key Features:**

*   **Control & Monitoring:** Access and manipulate steering, gas, brakes, and more.
*   **Real-time Data:** Read crucial data such as speed and steering angle.
*   **Open Source & Community Driven:** Contribute to the project and expand its capabilities.
*   **Extensive Documentation:** Comprehensive guides and resources for using, contributing, and extending opendbc.
*   **Supported Cars:** Wide support for vehicles with LKAS and ACC interfaces.

---

## Getting Started

Get up and running quickly with these simple steps:

```bash
git clone https://github.com/commaai/opendbc.git
cd opendbc

# Run all dependencies and tests
./test.sh
```

See the [`examples/`](examples/) directory for sample programs, including [`examples/joystick.py`](examples/joystick.py), which lets you control your car with a joystick.

## Core Components

*   [`opendbc/dbc/`](opendbc/dbc/): Stores DBC (Database CAN) files.
*   [`opendbc/can/`](opendbc/can/): Provides a library for parsing and building CAN messages from DBC files.
*   [`opendbc/car/`](opendbc/car/): Offers a high-level Python interface for interacting with your car.
*   [`opendbc/safety/`](opendbc/safety/): Implements functional safety features.

## How to Add Your Car (Porting)

Expand opendbc's reach by adding support for your vehicle. The process generally involves:

*   Connecting to the car with a comma 3X and a car harness.
*   Reverse Engineering CAN messages via `cabana`.
*   Creating `carstate.py`, `carcontroller.py`, and other necessary modules.
*   Tuning for optimal lateral and longitudinal control using tools like the [longitudinal maneuvers](https://github.com/commaai/openpilot/tree/master/tools/longitudinal_maneuvers).

Detailed instructions are available in the documentation.

## Contributing

Become a part of the opendbc community! Development is coordinated on GitHub and [Discord](https://discord.comma.ai).

### Roadmap

The project roadmap includes goals like improved ease-of-use, comprehensive type and line coverage, and expanded vehicle support.

## Safety Model

opendbc's safety features are tightly integrated with [openpilot](https://github.com/commaai/openpilot) and [panda](https://comma.ai/shop/panda) to ensure safe operation. The safety firmware enforces the principles of the [openpilot safety](https://github.com/commaai/openpilot/blob/master/docs/SAFETY.md) to ensure compliance.

## Code Quality and Rigor

The project implements stringent code quality checks to guarantee safety, including static analysis with [cppcheck](https://github.com/danmar/cppcheck/) (including MISRA C:2012 violations), strict compiler flags, and thorough unit and mutation testing.

## Bounties

We offer bounties for contributions, including:

*   $2000 - For any car brand/platform port.
*   $250 - For any car model port.
*   $300 - For reverse engineering a new Actuation Message.

See [comma.ai/bounties](comma.ai/bounties) for more details.

## Frequently Asked Questions (FAQ)

*   **How do I use this?** Requires a [comma 3X](https://comma.ai/shop/comma-3x).
*   **Which cars are supported?** Check the [supported cars list](docs/CARS.md).
*   **Can I add support for my car?** Yes! Find out how [here](https://github.com/commaai/opendbc/blob/docs/README.md#how-to-port-a-car).
*   **Which cars *can* be supported?** Any with LKAS and ACC. See [here](https://github.com/commaai/openpilot/blob/master/docs/CARS.md#dont-see-your-car-here).
*   **How does this work?** Replace your car's built-in lane keep and adaptive cruise features. See [this talk](https://www.youtube.com/watch?v=FL8CxUSfipM).
*   **Is there a timeline or roadmap for adding car support?** Car support is primarily community driven.

### Key Terms

Definitions of frequently used terms can be found in the README.

### More Resources

Explore these additional resources for deeper understanding and assistance:

*   [*How Do We Control The Car?*](https://www.youtube.com/watch?v=nNU6ipme878&pp=ygUoY29tbWEgY29uIDIwMjEgaG93IGRvIHdlIGNvbnRyb2wgdGhlIGNhcg%3D%3D) by [@robbederks](https://github.com/robbederks) from COMMA_CON 2021
*   [*How to Port a Car*](https://www.youtube.com/watch?v=XxPS5TpTUnI&t=142s&pp=ygUPamFzb24gY29tbWEgY29u) by [@jyoung8607](https://github.com/jyoung8607) from COMMA_CON 2023
*   [commaCarSegments](https://huggingface.co/datasets/commaai/commaCarSegments)
*   [cabana](https://github.com/commaai/openpilot/tree/master/tools/cabana#readme)
*   [can_print_changes.py](https://github.com/commaai/openpilot/blob/master/selfdrive/debug/can_print_changes.py)
*   [longitudinal maneuvers](https://github.com/commaai/openpilot/tree/master/tools/longitudinal_maneuvers)
*   [opendbc data](https://commaai.github.io/opendbc-data/)

## Join the Team - We're Hiring! [comma.ai/jobs](https://comma.ai/jobs)

comma.ai is looking for engineers to join our team and work on opendbc and [openpilot](https://github.com/commaai/openpilot).