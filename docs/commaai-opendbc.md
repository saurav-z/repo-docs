# opendbc: Your Python API for Automotive Control and Data

**opendbc empowers you to control and monitor your car's systems, providing a powerful Python API for accessing and manipulating vehicle data.**  Explore the [opendbc repository](https://github.com/commaai/opendbc) for more details.

### Key Features:

*   **Control:** Access and manipulate your car's steering, gas, and brakes.
*   **Data Acquisition:** Read vital information like speed and steering angle.
*   **Car Support:**  Designed to support a wide range of vehicles, particularly those with advanced driver-assistance systems (ADAS).
*   **Open Source:**  Built on open-source principles, fostering community contributions and innovation.
*   **Safety-Focused:** Rigorous code review and testing to ensure safe and reliable operation.
*   **Extendable:** Comprehensive documentation to support adding support for new cars.

###  Explore:

*   [**Documentation**](https://docs.comma.ai)
*   [**Contribute**](https://github.com/commaai/openpilot/blob/master/docs/CONTRIBUTING.md)
*   [**Discord Community**](https://discord.comma.ai)
*   [**License**](LICENSE)
*   [**X (Twitter)**](https://x.com/comma_ai)

---

opendbc targets vehicles equipped with electronically-actuated systems like Lane Keeping Assist (LKAS) and Adaptive Cruise Control (ACC), which are common in cars since 2016. The primary objective is to provide comprehensive support for controlling these features, with a broader vision of developing a complete vehicle management application.

## Quick Start

Get started with the following commands:

```bash
git clone https://github.com/commaai/opendbc.git
cd opendbc
./test.sh
```
*   **Installation:** `./test.sh` installs dependencies, compiles, lints, and runs tests. It's ideal for getting started.
*   **Individual Commands:**  Install dependencies using `pip3 install -e .[testing,docs]`, build with `scons -j8`, run tests with `pytest .`, and run the linter with `lefthook run lint`.

Explore example programs located in the [`examples/`](examples/) directory to read and control your car's state.  [`examples/joystick.py`](examples/joystick.py) is a great example.

### Project Structure

*   [`opendbc/dbc/`](opendbc/dbc/): Contains DBC files that define CAN messages.
*   [`opendbc/can/`](opendbc/can/): Provides a library to parse and build CAN messages from DBC files.
*   [`opendbc/car/`](opendbc/car/): Offers a high-level Python library for interfacing with cars.
*   [`opendbc/safety/`](opendbc/safety/): Implements functional safety features for supported cars.

## How to Port a Car

This guide details the process of adding support for new cars or enhancing existing ones.

### 1. Connect to the Car

The initial step involves connecting to your car using a comma 3X and a compatible car harness.

*   Harnesses:  You can purchase pre-designed harnesses at [comma.ai/shop](https://comma.ai/shop). If a harness isn't available, consider using a "developer harness" and attaching the appropriate connector.

### 2. Structure of a Port

Car ports are structured within `opendbc/car/<brand>/`, and generally include the following components:

*   `carstate.py`: Parses information from the CAN stream using the car's DBC file.
*   `carcontroller.py`: Outputs CAN messages to control the car.
*   `<brand>can.py`: Provides Python helpers based on the DBC file for building CAN messages.
*   `fingerprints.py`: Holds a database of ECU firmware versions for identifying car models.
*   `interface.py`: Offers a high-level class for interacting with the car.
*   `radar_interface.py`: Parses radar data.
*   `values.py`: Enumerates supported car models for the brand.

### 3. Reverse Engineer CAN messages

*   Record a route with interesting events, such as LKAS and ACC activation, and extensive steering maneuvers.
*   Load the recorded route into [cabana](https://github.com/commaai/openpilot/tree/master/tools/cabana) to analyze CAN messages.

### 4. Tuning

#### Longitudinal

*   Use the [longitudinal maneuvers](https://github.com/commaai/openpilot/tree/master/tools/longitudinal_maneuvers) report to evaluate and tune your car's longitudinal control.

## Contributing

Contributions are managed on GitHub and discussed in the [Discord](https://discord.comma.ai) server, particularly in the `#dev-opendbc-cars` channel and the `Vehicle Specific` section.

### Roadmap

**Near Term**

*   `pip install opendbc`
*   Achieve 100% type and line coverage.
*   Simplify car port creation through refactoring, tools, and enhanced documentation.
*   Improve the representation of all supported car states.

**Longer Term**

*   Expand support to all cars with LKAS + ACC interfaces.
*   Automated lateral and longitudinal control/tuning evaluation.
*   Implement auto-tuning for lateral and longitudinal control.
*   Develop Automatic Emergency Braking.

## Safety Model

The opendbc safety firmware, when used with a [panda](https://comma.ai/shop/panda), initially operates in `SAFETY_SILENT` mode, disabling CAN bus transmissions.  To send messages, you must select a safety mode.  Some modes (like `SAFETY_ALLOUTPUT`) are disabled in release firmwares and require a custom build.

Safety modes optionally support `controls_allowed`, allowing or blocking messages based on customizable states.

## Code Rigor

The opendbc safety firmware is designed for use with openpilot and the panda. The safety model ensures [openpilot safety](https://github.com/commaai/openpilot/blob/master/docs/SAFETY.md). Rigor is enforced to guarantee safe and reliable operation.

### Testing:

*   [CI regression tests](https://github.com/commaai/opendbc/actions) are in place.
*   Static code analysis by [cppcheck](https://github.com/danmar/cppcheck/) and a specific addon for [MISRA C:2012](https://misra.org.uk/) violations. See [current coverage](opendbc/safety/tests/misra/coverage_table).
*   Compiler flags `-Wall -Wextra -Wstrict-prototypes -Werror` are enforced.
*   Unit tests for each supported car variant are performed.

The above tests are themselves tested by:
* a [mutation test](opendbc/safety/tests/misra/test_mutation.py) on the MISRA coverage
* 100% line coverage enforced on the safety unit tests

Additional testing is done with:
* the [ruff linter](https://github.com/astral-sh/ruff)
* and [mypy](https://mypy-lang.org/)

### Bounties

*   $2000 - [Any car brand / platform port](https://github.com/orgs/commaai/projects/26/views/1?pane=issue&itemId=47913774)
*   $250 - [Any car model port](https://github.com/orgs/commaai/projects/26/views/1?pane=issue&itemId=47913790)
*   $300 - [Reverse Engineering a new Actuation Message](https://github.com/orgs/commaai/projects/26/views/1?pane=issue&itemId=73445563)

For additional information, go to [comma.ai/bounties](comma.ai/bounties).

## FAQ

*   **How do I use this?**  A [comma 3X](https://comma.ai/shop/comma-3x) is designed to be the optimal platform for running and developing with opendbc and openpilot.

*   **Which cars are supported?**  Refer to the [supported cars list](docs/CARS.md).

*   **Can I add support for my car?**  Yes.  Community contributions drive most car support.  See the [car port guide](https://github.com/commaai/opendbc/blob/docs/README.md#how-to-port-a-car).

*   **Which cars can be supported?**  Cars with LKAS and ACC are good candidates. More details are available [here](https://github.com/commaai/openpilot/blob/master/docs/CARS.md#dont-see-your-car-here).

*   **How does this work?**  This system uses hardware to replace a car's built-in lane keeping and adaptive cruise features.  See [this talk](https://www.youtube.com/watch?v=FL8CxUSfipM).

*   **Is there a timeline or roadmap for adding car support?** Community contributions drive car support, with comma performing safety and quality validation. The more complete and popular a community port is, the more likely it will be validated.

### Terms

*   **port:** Support for a specific car
*   **lateral control:** Steering control
*   **longitudinal control:** Gas/brake control
*   **fingerprinting:** Automatic car identification
*   **[LKAS](https://en.wikipedia.org/wiki/Lane_departure_warning_system)**: Lane Keeping Assist
*   **[ACC](https://en.wikipedia.org/wiki/Adaptive_cruise_control)**: Adaptive Cruise Control
*   **[harness](https://comma.ai/shop/car-harness)**: Car-specific hardware for ADAS message interception
*   **[panda](https://github.com/commaai/panda)**: Hardware for CAN bus access
*   **[ECU](https://en.wikipedia.org/wiki/Electronic_control_unit)**: Vehicle computers
*   **[CAN bus](https://en.wikipedia.org/wiki/CAN_bus)**: Connects ECUs
*   **[cabana](https://github.com/commaai/openpilot/tree/master/tools/cabana#readme)**: CAN message reverse engineering tool
*   **[DBC file](https://en.wikipedia.org/wiki/CAN_bus#DBC)**: CAN bus message definitions
*   **[openpilot](https://github.com/commaai/openpilot)**: ADAS system for opendbc-supported cars
*   **[comma](https://github.com/commaai)**: The company behind opendbc
*   **[comma 3X](https://comma.ai/shop/comma-3x)**: Hardware for openpilot

### Resources

*   [*How Do We Control The Car?*](https://www.youtube.com/watch?v=nNU6ipme878&pp=ygUoY29tbWEgY29uIDIwMjEgaG93IGRvIHdlIGNvbnRyb2wgdGhlIGNhcg%3D%3D) by [@robbederks](https://github.com/robbederks) from COMMA_CON 2021
*   [*How to Port a Car*](https://www.youtube.com/watch?v=XxPS5TpTUnI&t=142s&pp=ygUPamFzb24gY29tbWEgY29u) by [@jyoung8607](https://github.com/jyoung8607) from COMMA_CON 2023
*   [commaCarSegments](https://huggingface.co/datasets/commaai/commaCarSegments): CAN data dataset
*   [cabana](https://github.com/commaai/openpilot/tree/master/tools/cabana#readme): CAN message reverse engineering
*   [can_print_changes.py](https://github.com/commaai/openpilot/blob/master/selfdrive/debug/can_print_changes.py): Diff CAN bus
*   [longitudinal maneuvers](https://github.com/commaai/openpilot/tree/master/tools/longitudinal_maneuvers): Tuning tool
*   [opendbc data](https://commaai.github.io/opendbc-data/): Longitudinal maneuver evaluations

## Join the Team: [comma.ai/jobs](https://comma.ai/jobs)

comma is seeking engineers to contribute to opendbc and [openpilot](https://github.com/commaai/openpilot). Contributions are welcomed.