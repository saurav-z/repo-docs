<div align="center" style="text-align: center;">

<h1>opendbc</h1>
<p>
  <b>Unlock Your Car's Potential: Take Control of Steering, Gas, and Brakes with opendbc!</b>
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

## opendbc: Your Python API for Advanced Car Control

opendbc is a powerful Python API that allows you to read and control your car's systems, including steering, gas, brakes, and more. Built for the openpilot project, opendbc provides a comprehensive platform for accessing and manipulating vehicle data.  Explore the capabilities of opendbc and its role in enabling cutting-edge automotive applications.  Learn more and contribute to the project on [GitHub](https://github.com/commaai/opendbc).

**Key Features:**

*   **Comprehensive Car Control:** Interface with steering, gas, and brakes.
*   **Real-time Data Access:** Read speed, steering angle, and other vital vehicle data.
*   **DBC File Integration:** Utilizes DBC files for parsing and building CAN messages.
*   **Car Porting Support:** Contribute and extend support to new vehicles with detailed documentation and guidance.
*   **Safety Focused:** Rigorous safety model and code rigor for reliable operation.
*   **Community-Driven:** Actively developed and supported by a vibrant community on Discord.

## Getting Started

Quickly get up and running with opendbc:

```bash
git clone https://github.com/commaai/opendbc.git
cd opendbc

# Install dependencies, build, run tests, and lint
./test.sh
```

Explore the `examples/` directory for practical demonstrations of how to read car state and control vehicle functions, including a joystick control example.

## Project Structure

*   **`opendbc/dbc/`**: Repository for DBC files, defining CAN message structures.
*   **`opendbc/can/`**: Library for parsing and constructing CAN messages from DBC files.
*   **`opendbc/car/`**: High-level Python library for interacting with car systems.
*   **`opendbc/safety/`**: Ensures the safety of all cars supported by opendbc/car/.

## How to Port a Car

Follow these steps to add support for your car. The process involves connecting to the car, reverse engineering CAN messages, and tuning control parameters.  Detailed information can be found in the original README and is summarized below:

1.  **Connect to the Car:** Use a comma 3X device and car harness.
2.  **Structure of a Port:** Create necessary files within the `opendbc/car/<brand>/` directory for car-specific functionalities.
3.  **Reverse Engineer CAN Messages:** Use tools like cabana to analyze and understand CAN data.
4.  **Tuning:** Utilize tools like longitudinal maneuvers to tune car control.

## Contributing

Contribute to opendbc development through GitHub and the Discord community. Check out the #dev-opendbc-cars channel.

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

## Safety Model

The opendbc safety firmware ensures safe operation when used with openpilot and the panda hardware, with a [strong focus on code rigor](opendbc/safety/tests/misra/coverage_table).

## Bounties

Earn bounties for your contributions. Check for current bounties at [comma.ai/bounties](comma.ai/bounties).

## FAQ

*   **How do I use this?** A [comma 3X](https://comma.ai/shop/comma-3x) is the best way to run opendbc and openpilot.
*   **Which cars are supported?** See the [supported cars list](docs/CARS.md).
*   **Can I add support for my car?** Yes, read the [guide](https://github.com/commaai/opendbc/blob/docs/README.md#how-to-port-a-car).
*   **Which cars can be supported?** Any car with LKAS and ACC. More info [here](https://github.com/commaai/openpilot/blob/master/docs/CARS.md#dont-see-your-car-here).
*   **How does this work?** It replaces your car's built-in lane keep and adaptive cruise features. See [this talk](https://www.youtube.com/watch?v=FL8CxUSfipM) for an in-depth explanation.
*   **Is there a timeline or roadmap for adding car support?** No, most car support comes from the community.

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

### More resources

*   [*How Do We Control The Car?*](https://www.youtube.com/watch?v=nNU6ipme878&pp=ygUoY29tbWEgY29uIDIwMjEgaG93IGRvIHdlIGNvbnRyb2wgdGhlIGNhcg%3D%3D) by [@robbederks](https://github.com/robbederks) from COMMA_CON 2021
*   [*How to Port a Car*](https://www.youtube.com/watch?v=XxPS5TpTUnI&t=142s&pp=ygUPamFzb24gY29tbWEgY29u) by [@jyoung8607](https://github.com/jyoung8607) from COMMA_CON 2023
*   [commaCarSegments](https://huggingface.co/datasets/commaai/commaCarSegments): a massive dataset of CAN data from 300 different car models
*   [cabana](https://github.com/commaai/openpilot/tree/master/tools/cabana#readme): our tool for reverse engineering CAN messages
*   [can_print_changes.py](https://github.com/commaai/openpilot/blob/master/selfdrive/debug/can_print_changes.py): diff the whole CAN bus across two drives, such as one without any LKAS and one with LKAS
*   [longitudinal maneuvers](https://github.com/commaai/openpilot/tree/master/tools/longitudinal_maneuvers): a tool for evaluating and tuning longitudinal control
*   [opendbc data](https://commaai.github.io/opendbc-data/): a repository of longitudinal maneuver evaluations

## Come work with us -- [comma.ai/jobs](https://comma.ai/jobs)

comma is hiring engineers to work on opendbc and [openpilot](https://github.com/commaai/openpilot). We love hiring contributors.