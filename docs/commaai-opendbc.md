# opendbc: Your Python API for Car Control and Data Access

**Control your car's steering, gas, and brakes, and access real-time data with opendbc – a powerful Python API.**  Explore the official [opendbc GitHub repository](https://github.com/commaai/opendbc) for more details.

*   **Control & Read:** Actuate gas, brakes, and steering, and read crucial vehicle data like speed and steering angle.
*   **ADAS Integration:** Designed primarily for [openpilot](https://github.com/commaai/openpilot), extendable to many car brands.
*   **Extensive Support:**  Read and write a variety of vehicle functions, including EV charging status and door locks.
*   **Community Driven:** Join the community on [Discord](https://discord.comma.ai) to collaborate and contribute.
*   **Safety Focused:** Rigorous code standards, including MISRA C:2012 compliance, are applied to the safety firmware.

## Key Features

*   **DBC File Repository:** Includes a comprehensive collection of DBC files for various vehicles.
*   **CAN Message Handling:** Built-in libraries for parsing and constructing CAN messages.
*   **High-Level Car Interface:** User-friendly Python library for seamless car interaction.
*   **Safety Model:** Functional safety features for all supported cars, with multiple safety modes.
*   **Extensive Testing:** Robust CI regression tests, including static code analysis and unit tests.
*   **Community & Open Source:** Actively seeking contributions to expand car support and features.

## Getting Started

Get up and running quickly:

```bash
git clone https://github.com/commaai/opendbc.git
cd opendbc
./test.sh # Installs dependencies, builds, lints, and runs tests
```

Explore example programs in the [`examples/`](examples/) directory, such as [`examples/joystick.py`](examples/joystick.py) to control a car with a joystick.

## Project Structure

*   [`opendbc/dbc/`](opendbc/dbc/): Stores [DBC](https://en.wikipedia.org/wiki/CAN_bus#DBC) files.
*   [`opendbc/can/`](opendbc/can/): Library for parsing and building CAN messages from DBC files.
*   [`opendbc/car/`](opendbc/car/): High-level Python library for interfacing with cars.
*   [`opendbc/safety/`](opendbc/safety/): Functional safety for supported cars.

## Adding Support for Your Car

Expand your car's control and data access capabilities. Learn to add support for new vehicles by connecting to the car, reverse engineering CAN messages, and tuning your new car configuration. Detailed instructions are available to get you started.

## Contribute

Contribute to opendbc's development on [GitHub](https://github.com/commaai/opendbc) and [Discord](https://discord.comma.ai). Check out the `#dev-opendbc-cars` channel for help.

### Roadmap

*   [ ]  `pip install opendbc`
*   [ ]  100% type coverage
*   [ ]  100% line coverage
*   [ ]  Make car ports easier: refactors, tools, tests, and docs
*   [ ]  Expose the state of all supported cars better: https://github.com/commaai/opendbc/issues/1144
*   [ ]  Extend support to every car with LKAS + ACC interfaces.
*   [ ]  Automatic lateral and longitudinal control/tuning evaluation.
*   [ ]  Auto-tuning for lateral and longitudinal control.
*   [ ]  Automatic Emergency Braking.

## Safety Model

The `opendbc` safety firmware, designed for use with [openpilot](https://github.com/commaai/openpilot) and [panda](https://comma.ai/shop/panda), operates in various safety modes. These modes ensure a secure environment for vehicle control.

## Code Rigor

The project maintains high standards for code quality, including MISRA C:2012 compliance, thorough testing, and strict compiler flags, ensuring reliability and safety.

## Bounties

Contribute and get rewarded! Learn more about available bounties for car porting and reverse engineering on [comma.ai/bounties](comma.ai/bounties).

## FAQ

**Q: How do I use this?**

A: The [comma 3X](https://comma.ai/shop/comma-3x) is custom-designed to be the best way to run and develop opendbc and openpilot.

**Q: Which cars are supported?**

A: See the [supported cars list](docs/CARS.md).

**Q: Can I add support for my car?**

A: Yes! Most car support comes from the community. Read the guide [here](https://github.com/commaai/opendbc/blob/docs/README.md#how-to-port-a-car).

**Q: Which cars can be supported?**

A: Any car with LKAS and ACC. More info [here](https://github.com/commaai/openpilot/blob/master/docs/CARS.md#dont-see-your-car-here).

**Q: How does this work?**

A: We designed hardware to replace your car's built-in lane keep and adaptive cruise features. See [this talk](https://www.youtube.com/watch?v=FL8CxUSfipM) for an in-depth explanation.

**Q: Is there a timeline or roadmap for adding car support?**

A: Most car support comes from the community, with comma doing final safety and quality validation.

### Terms

*   **port**: The integration and support of a specific car.
*   **lateral control**: Steering control.
*   **longitudinal control**: Gas/brakes control.
*   **fingerprinting**: Automatic process for identifying the car.
*   **[LKAS](https://en.wikipedia.org/wiki/Lane_departure_warning_system)**: Lane keeping assist.
*   **[ACC](https://en.wikipedia.org/wiki/Adaptive_cruise_control)**: Adaptive cruise control.
*   **[harness](https://comma.ai/shop/car-harness)**: Car-specific hardware for ADAS message interception.
*   **[panda](https://github.com/commaai/panda)**: Hardware for accessing a car's CAN bus.
*   **[ECU](https://en.wikipedia.org/wiki/Electronic_control_unit)**: Electronic control modules inside the car.
*   **[CAN bus](https://en.wikipedia.org/wiki/CAN_bus)**: A bus connecting ECUs.
*   **[cabana](https://github.com/commaai/openpilot/tree/master/tools/cabana#readme)**: Tool for reverse engineering CAN messages.
*   **[DBC file](https://en.wikipedia.org/wiki/CAN_bus#DBC)**: Defines messages on a CAN bus.
*   **[openpilot](https://github.com/commaai/openpilot)**: An ADAS system supported by opendbc.
*   **[comma](https://github.com/commaai)**: The company behind opendbc.
*   **[comma 3X](https://comma.ai/shop/comma-3x)**: Hardware used to run openpilot.

### More Resources

*   [*How Do We Control The Car?*](https://www.youtube.com/watch?v=nNU6ipme878&pp=ygUoY29tbWEgY29uIDIwMjEgaG93IGRvIHdlIGNvbnRyb2wgdGhlIGNhcg%3D%3D) by [@robbederks](https://github.com/robbederks) from COMMA_CON 2021
*   [*How to Port a Car*](https://www.youtube.com/watch?v=XxPS5TpTUnI&t=142s&pp=ygUPamFzb24gY29tbWEgY29u) by [@jyoung8607](https://github.com/jyoung8607) from COMMA_CON 2023
*   [commaCarSegments](https://huggingface.co/datasets/commaai/commaCarSegments): A massive dataset of CAN data from 300 different car models.
*   [cabana](https://github.com/commaai/openpilot/tree/master/tools/cabana#readme): Our tool for reverse engineering CAN messages.
*   [can_print_changes.py](https://github.com/commaai/openpilot/blob/master/selfdrive/debug/can_print_changes.py): Diff the whole CAN bus across two drives.
*   [longitudinal maneuvers](https://github.com/commaai/openpilot/tree/master/tools/longitudinal_maneuvers): Tool for evaluating and tuning longitudinal control.
*   [opendbc data](https://commaai.github.io/opendbc-data/): A repository of longitudinal maneuver evaluations

## Join the Team

Explore career opportunities and contribute to the future of automotive technology. Join comma! [comma.ai/jobs](https://comma.ai/jobs)