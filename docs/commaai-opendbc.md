# opendbc: Your Python API for Car Control and Data Access

**opendbc empowers you to control your car's steering, gas, and brakes, while also reading vital data like speed and steering angle, all through a user-friendly Python API.**  [Explore the opendbc repository on GitHub](https://github.com/commaai/opendbc).

*   **Comprehensive Car Control:** Take control of steering, acceleration, and braking systems.
*   **Real-Time Data Acquisition:** Read critical vehicle data such as speed, steering angle, and more.
*   **DBC File Integration:** Uses DBC files for CAN bus message parsing.
*   **Car Porting Guide:** Detailed instructions and resources for adding support for new car models.
*   **Safety Focused:** Rigorous code rigor and safety model to ensure reliability.

---

opendbc provides a powerful Python API for interfacing with your vehicle's systems.  It is designed to interact with the car's internal networks to read data and control various features, particularly for cars with electronic control systems.

**Key Features:**

*   **Control and Read:**  Interact with your car's systems by controlling actuation and retrieving data from the CAN bus.
*   **DBC Support:** Utilizes DBC files to decode and work with CAN messages.
*   **Car Porting Guide:** Provides guidance on supporting new car models.
*   **Safety First:** Dedicated safety firmware to enforce safe operation.
*   **Community Driven:** Leverages contributions from the open-source community.

**Getting Started:**

```bash
git clone https://github.com/commaai/opendbc.git
cd opendbc
./test.sh # Installs dependencies, builds, tests, and lints
```

The `examples/` directory includes example programs for reading car state and controlling steering, gas, and brakes.  For example, `examples/joystick.py` lets you control a car with a joystick.

**Project Structure:**

*   `opendbc/dbc/`: Stores DBC files for different car models.
*   `opendbc/can/`:  Library for parsing and constructing CAN messages from DBC files.
*   `opendbc/car/`:  High-level Python library for car interfaces.
*   `opendbc/safety/`: Safety-critical code and firmware.

**How to Port a Car**

Support for new cars is a community effort. The goal is to control steering, gas, and brakes on the supported cars.  Adding support typically involves:

1.  Connecting to the car using a comma 3X and car harness.
2.  Creating a car port within the `opendbc/car/<brand>/` directory, consisting of files to handle car state, control, CAN message construction, and model identification.
3.  Reverse engineering CAN messages using a tool like cabana.
4.  Tuning for optimal longitudinal control.

**Contributing**

*   All development happens on GitHub and [Discord](https://discord.comma.ai). Check out the `#dev-opendbc-cars` channel and `Vehicle Specific` section.
*   We welcome contributions towards extending support to every car with LKAS + ACC interfaces.

**Safety Model**

opendbc's safety firmware, used with [openpilot](https://github.com/commaai/openpilot) and [panda](https://comma.ai/shop/panda), ensures safe operation. It enforces safety protocols, and the application code undergoes thorough testing, including static analysis, MISRA checks, and unit tests, to uphold high standards of code quality.

**Bounties**

Earn rewards for contributing to opendbc! Bounties are available for adding support for new cars, reverse engineering messages, and more. See [comma.ai/bounties](https://comma.ai/bounties) for details.

**FAQ**

*   **How do I use this?**  A [comma 3X](https://comma.ai/shop/comma-3x) is recommended to run and develop opendbc.
*   **Which cars are supported?**  See the [supported cars list](docs/CARS.md).
*   **Can I add support for my car?**  Yes, see the guide [here](https://github.com/commaai/opendbc/blob/docs/README.md#how-to-port-a-car).
*   **Which cars can be supported?**  Any car with LKAS and ACC. More info [here](https://github.com/commaai/openpilot/blob/master/docs/CARS.md#dont-see-your-car-here).
*   **How does this work?**  We designed hardware to replace your car's built-in lane keep and adaptive cruise features.
*   **Is there a timeline or roadmap for adding car support?**  Most car support comes from the community.

**Terms**

*   **Port**: integration and support of a specific car
*   **Lateral control**: aka steering control
*   **Longitudinal control**: aka gas/brakes control
*   **Fingerprinting**: automatic process for identifying the car
*   **LKAS**: lane keeping assist
*   **ACC**: adaptive cruise control
*   **Harness**: car-specific hardware to attach to the car
*   **Panda**: hardware used to get on a car's CAN bus
*   **ECU**: computers or control modules inside the car
*   **CAN bus**: a bus that connects the ECUs in a car
*   **Cabana**: tool for reverse engineering CAN messages
*   **DBC file**: contains definitions for messages on a CAN bus
*   **openpilot**: an ADAS system for cars supported by opendbc
*   **Comma**: the company behind opendbc
*   **Comma 3X**: the hardware used to run openpilot

**More Resources**

*   [*How Do We Control The Car?*](https://www.youtube.com/watch?v=nNU6ipme878&pp=ygUoY29tbWEgY29uIDIwMjEgaG93IGRvIHdlIGNvbnRyb2wgdGhlIGNhcg%3D%3D) by [@robbederks](https://github.com/robbederks) from COMMA_CON 2021
*   [*How to Port a Car*](https://www.youtube.com/watch?v=XxPS5TpTUnI&t=142s&pp=ygUPamFzb24gY29tbWEgY29u) by [@jyoung8607](https://github.com/jyoung8607) from COMMA_CON 2023
*   [commaCarSegments](https://huggingface.co/datasets/commaai/commaCarSegments): a massive dataset of CAN data from 300 different car models
*   [cabana](https://github.com/commaai/openpilot/tree/master/tools/cabana#readme): our tool for reverse engineering CAN messages
*   [can_print_changes.py](https://github.com/commaai/openpilot/blob/master/selfdrive/debug/can_print_changes.py): diff the whole CAN bus across two drives
*   [longitudinal maneuvers](https://github.com/commaai/openpilot/tree/master/tools/longitudinal_maneuvers): a tool for evaluating and tuning longitudinal control
*   [opendbc data](https://commaai.github.io/opendbc-data/): a repository of longitudinal maneuver evaluations

**Join the Team**

[comma.ai/jobs](https://comma.ai/jobs) is hiring engineers to work on opendbc and [openpilot](https://github.com/commaai/openpilot).