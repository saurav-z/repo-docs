# opendbc: Your Python API for Automotive Control and Data Access

**Take control of your car's gas, brakes, and steering with opendbc, a powerful Python API designed for automotive enthusiasts and developers.** Learn more and contribute on the [original repo](https://github.com/commaai/opendbc).

## Key Features

*   **Control & Read Data:** Access and control vehicle systems like steering, gas, brakes, speed, and steering angle.
*   **Extensive Car Support:** Designed to support a wide range of vehicles with LKAS (Lane Keeping Assist) and ACC (Adaptive Cruise Control) systems.
*   **Open Source & Community Driven:**  Leverage the power of open-source development and join the active [Discord](https://discord.comma.ai) community.
*   **Well-Documented:** Comprehensive documentation, including a [supported cars list](docs/CARS.md), guides, and examples.
*   **Safety Focused:** Rigorous code rigor with a focus on safety, including MISRA C:2012 compliance and extensive testing.
*   **Active Development:** Continuous improvement and new features, including support for automatic tuning and expansion to more vehicles.

## Quick Start

```bash
git clone https://github.com/commaai/opendbc.git
cd opendbc

# Run the all-in-one script for setup, building, linting, and testing.
./test.sh

# You can also run the commands individually
pip3 install -e .[testing,docs]  # install dependencies
scons -j8                        # build with 8 cores
pytest .                         # run the tests
lefthook run lint                # run the linter
```

## Project Structure

*   [`opendbc/dbc/`](opendbc/dbc/) - Houses [DBC](https://en.wikipedia.org/wiki/CAN_bus#DBC) files, which define the CAN bus messages.
*   [`opendbc/can/`](opendbc/can/) - A library for parsing and building CAN messages from DBC files.
*   [`opendbc/car/`](opendbc/car/) -  A high-level Python library for interacting with cars.
*   [`opendbc/safety/`](opendbc/safety/) -  Ensures the safety of vehicle control functions, particularly for vehicles running openpilot.

## Contributing

All development is coordinated on GitHub and [Discord](https://discord.comma.ai). Check out the `#dev-opendbc-cars` channel and `Vehicle Specific` section to get involved.

### Roadmap

*   **Short Term:** `pip install opendbc`, 100% type and line coverage, and improved car port tools and documentation.
*   **Longer Term:** Expand support to all LKAS + ACC cars, implement auto-tuning for lateral and longitudinal control, and automatic emergency braking (AEB) integration.

## Safety Model

opendbc, used with [openpilot](https://github.com/commaai/openpilot) and [panda](https://comma.ai/shop/panda), enforces a safety model. By default, the CAN buses are silent until a safety mode is selected. This can allow or block certain message transmissions based on the situation. 

## Code Rigor

opendbc uses rigorous testing and code analysis:

*   Static code analysis with [cppcheck](https://github.com/danmar/cppcheck/), including MISRA C:2012 checks.
*   Strict compiler flags (`-Wall -Wextra -Wstrict-prototypes -Werror`).
*   Unit tests for safety logic with 100% line coverage and mutation testing.
*   Linters ([ruff](https://github.com/astral-sh/ruff)) and type checking ([mypy](https://mypy-lang.org/)) for the car interface library.

## Bounties

Earn rewards for contributing:
*   **$2000** - New car brand/platform port
*   **$250** - New car model port
*   **$300** - Reverse Engineering a new Actuation Message
*   More at [comma.ai/bounties](https://comma.ai/bounties).

## FAQ

**Q: How do I use opendbc?**
**A:**  A [comma 3X](https://comma.ai/shop/comma-3x) is the recommended hardware for running opendbc and openpilot.

**Q: Which cars are supported?**
**A:** See the [supported cars list](docs/CARS.md).

**Q: Can I add support for my car?**
**A:** Yes, the community is the primary driver of new car support. See the car port guide [here](https://github.com/commaai/opendbc/blob/docs/README.md#how-to-port-a-car).

**Q: Which cars can be supported?**
**A:** Any car with LKAS and ACC. More info [here](https://github.com/commaai/openpilot/blob/master/docs/CARS.md#dont-see-your-car-here).

**Q: How does this work?**
**A:**  Hardware replaces the car's built-in lane keep and adaptive cruise features. Watch [this talk](https://www.youtube.com/watch?v=FL8CxUSfipM) for details.

**Q: Is there a roadmap for adding car support?**
**A:**  Car support is primarily community-driven, with comma providing final validation. The more complete the car port and the more popular the car, the more likely it is to be prioritized.

## Terms

*   **port**: Support of a specific car.
*   **lateral control**: Steering control.
*   **longitudinal control**: Gas/brakes control.
*   **fingerprinting**: Automatic car identification.
*   **[LKAS](https://en.wikipedia.org/wiki/Lane_departure_warning_system)**: Lane Keeping Assist.
*   **[ACC](https://en.wikipedia.org/wiki/Adaptive_cruise_control)**: Adaptive Cruise Control.
*   **[harness](https://comma.ai/shop/car-harness)**: Car-specific hardware to connect to the car.
*   **[panda](https://github.com/commaai/panda)**: Hardware for CAN bus access.
*   **[ECU](https://en.wikipedia.org/wiki/Electronic_control_unit)**: Car's computer modules.
*   **[CAN bus](https://en.wikipedia.org/wiki/CAN_bus)**: Car's internal communication system.
*   **[cabana](https://github.com/commaai/openpilot/tree/master/tools/cabana#readme)**: Tool to reverse engineer CAN messages.
*   **[DBC file](https://en.wikipedia.org/wiki/CAN_bus#DBC)**: Defines CAN messages.
*   **[openpilot](https://github.com/commaai/openpilot)**: ADAS system supported by opendbc.
*   **[comma](https://github.com/commaai)**: The company behind opendbc.
*   **[comma 3X](https://comma.ai/shop/comma-3x)**: The hardware used to run openpilot.

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