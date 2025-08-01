# opendbc: Your Python API for Car Control

**Control your car's steering, gas, and brakes with opendbc, a powerful Python API for accessing and manipulating vehicle data, supporting advanced driver-assistance systems (ADAS) features.**

[View the opendbc repository on GitHub](https://github.com/commaai/opendbc)

## Key Features:

*   **Complete Car Control:** Interface with your car's steering, gas, and brakes.
*   **Data Acquisition:** Read real-time data like speed and steering angle.
*   **ADAS Focus:** Designed to support and enhance features like Lane Keeping Assist (LKAS) and Adaptive Cruise Control (ACC).
*   **Extensive Documentation:** Comprehensive guides and examples to help you get started and contribute.
*   **Community Driven:** Benefit from an active community on Discord and GitHub.
*   **Safety-Focused:** Rigorous code rigor and safety model for reliable operation.

## Quick Start:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/commaai/opendbc.git
    cd opendbc
    ```
2.  **Run the test script (installs dependencies, builds, and runs tests):**
    ```bash
    ./test.sh
    ```

3.  **Explore the examples:**
    *   [`examples/`](examples/) provides sample programs.
    *   [`examples/joystick.py`](examples/joystick.py) lets you control a car with a joystick.

## Project Structure:

*   [`opendbc/dbc/`](opendbc/dbc/): Contains DBC (Database CAN) files.
*   [`opendbc/can/`](opendbc/can/): Library for parsing and building CAN messages.
*   [`opendbc/car/`](opendbc/car/): High-level Python library for interfacing with cars.
*   [`opendbc/safety/`](opendbc/safety/): Functional safety components for supported cars.

## How to Port a Car

Follow these steps to add support for your car.
1. Connect to the Car.
2. Reverse engineer CAN messages by recording a route with interesting events and load up that route in Cabana.
3. Tune and add the following:
    *   `carstate.py`: parses the relevant information from the CAN stream.
    *   `carcontroller.py`: outputs CAN messages to control the car.
    *   `<brand>can.py`: Python helpers to build CAN messages.
    *   `fingerprints.py`: database of ECU firmware versions.
    *   `interface.py`: high-level class for car interfacing.
    *   `radar_interface.py`: parses the radar.
    *   `values.py`: enumerates the brand's supported cars.

## Contributing

Join the opendbc community on GitHub and [Discord](https://discord.comma.ai) to contribute and collaborate.

### Roadmap

*   [ ]  Implement `pip install opendbc`.
*   [ ]  Achieve 100% type and line coverage.
*   [ ]  Improve car port development with refactors, tools, tests, and docs.
*   [ ]  Improve state exposure of supported cars.

## Safety Model

Opendbc's safety firmware, when used with a [panda](https://comma.ai/shop/panda), operates in `SAFETY_SILENT` mode by default, ensuring a safe starting point. Safety modes can optionally support `controls_allowed`, which allows or blocks a subset of messages based on a customizable state in the board.

## Code Rigor

The project's safety firmware adheres to strict code standards and is tested with unit tests and mutation tests.  

## Bounties

Earn rewards for contributing to opendbc:

*   $2000 - For any car brand / platform port
*   $250 - For any car model port
*   $300 - For reverse engineering a new Actuation Message

More bounties are available, especially for popular cars, at [comma.ai/bounties](comma.ai/bounties).

## FAQ

*   **How do I use this?** A [comma 3X](https://comma.ai/shop/comma-3x) is custom-designed to be the best way to run and develop opendbc and openpilot.
*   **Which cars are supported?** See the [supported cars list](docs/CARS.md).
*   **Can I add support for my car?** Yes, read the guide [here](https://github.com/commaai/opendbc/blob/docs/README.md#how-to-port-a-car).
*   **Which cars can be supported?** Any car with LKAS and ACC. More info [here](https://github.com/commaai/openpilot/blob/master/docs/CARS.md#dont-see-your-car-here).
*   **How does this work?** See [this talk](https://www.youtube.com/watch?v=FL8CxUSfipM) for an in-depth explanation.
*   **Is there a timeline or roadmap for adding car support?** No, most car support comes from the community, with comma doing final safety and quality validation.

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

Join comma's team of engineers working on opendbc and [openpilot](https://github.com/commaai/openpilot) – we welcome contributors!