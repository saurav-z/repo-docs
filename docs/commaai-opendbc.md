<div align="center" style="text-align: center;">

<h1>opendbc</h1>
<p>
  <b>opendbc: Your Python API to unlock advanced control of your car.</b>
  <br>
  Take control of steering, acceleration, braking, and more!
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

opendbc is a powerful Python API designed to give you control over your car's systems. Whether you're interested in advanced driver-assistance systems (ADAS) or building a comprehensive vehicle management app, opendbc provides the tools you need.  **[Explore the opendbc repository on GitHub](https://github.com/commaai/opendbc)**.

**Key Features:**

*   **Control & Read**: Read and write data to your car's systems: gas, brakes, steering, speed, steering angle, and more.
*   **ADAS Support**:  Primarily supports ADAS interfaces for openpilot.
*   **Extensive Support**:  Read and write as many vehicle functions as possible (EV charging, door locks, etc.).
*   **DBC Files**:  Uses DBC files to define and parse CAN messages, the common language of modern vehicles.
*   **Car Porting Guide**: Detailed documentation on how to add support for new car models.
*   **Safety Focused**: Built with safety in mind for use with openpilot and panda hardware.
*   **Community Driven**: Built with a focus on community contributions and development.

---

## Getting Started

### Installation & Running Tests

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/commaai/opendbc.git
    cd opendbc
    ```
2.  **Run the all-in-one script:**
    ```bash
    ./test.sh
    ```

    This script handles:
    * Dependency installation
    * Compilation
    * Linting
    * Testing

3.  **Individual Commands (if you prefer):**
    ```bash
    pip3 install -e .[testing,docs]  # install dependencies
    scons -j8                        # build with 8 cores
    pytest .                         # run the tests
    lefthook run lint                # run the linter
    ```
4.  **Examples**:  Find example programs in the `examples/` directory to get started.

### Project Structure

*   `opendbc/dbc/`: Stores [DBC](https://en.wikipedia.org/wiki/CAN_bus#DBC) files that define CAN messages.
*   `opendbc/can/`:  Provides a library for parsing and building CAN messages from DBC files.
*   `opendbc/car/`: Offers a high-level Python library for interfacing with cars.
*   `opendbc/safety/`: Implements functional safety measures for supported vehicles.

## How to Port a Car

This guide explains how to add support for new cars or enhance existing support. See the full documentation [here](https://github.com/commaai/opendbc/blob/docs/README.md#how-to-port-a-car).

### 1. Connect to the Car

*   Use a [comma 3X](https://comma.ai/shop/comma-3x) and a car harness (or a developer harness if one isn't available).

### 2. Structure of a port

*   The entirety of a car port lives in `opendbc/car/<brand>/`:
    *   `carstate.py`: parses relevant CAN data.
    *   `carcontroller.py`: outputs control messages.
    *   `<brand>can.py`: helper functions for building CAN messages.
    *   `fingerprints.py`: Identifies car models based on ECU firmware.
    *   `interface.py`: High-level class for interacting with the car.
    *   `radar_interface.py`: Parses radar data (if applicable).
    *   `values.py`: Defines supported car models for the brand.

### 3. Reverse Engineer CAN messages

*   Record a route with diverse events (LKAS, ACC, steering).
*   Load the recording into [cabana](https://github.com/commaai/openpilot/tree/master/tools/cabana) to analyze the data.

### 4. Tuning

*   Use the [longitudinal maneuvers](https://github.com/commaai/openpilot/tree/master/tools/longitudinal_maneuvers) report to tune longitudinal control.

## Contributing

Contributions are welcome! Coordinate development on [GitHub](https://github.com/commaai/opendbc) and [Discord](https://discord.comma.ai). Check out the `#dev-opendbc-cars` channel and `Vehicle Specific` section.

### Roadmap

*   **Short-Term Goals:**
    *   `pip install opendbc`
    *   100% type and line coverage.
    *   Enhance car porting: refactoring, tools, and docs.
    *   Improve the display of supported car states.

*   **Long-Term Goals:**
    *   Expand support to every car with LKAS and ACC.
    *   Automated lateral and longitudinal control/tuning evaluation.
    *   Auto-tuning for lateral and longitudinal control.
    *   Develop Automatic Emergency Braking capabilities.

## Safety Model

opendbc's safety firmware, in conjunction with [openpilot](https://github.com/commaai/openpilot) and [panda](https://comma.ai/shop/panda), ensures safe operation.

### Code Rigor

The safety firmware code undergoes rigorous testing:

*   Static code analysis (cppcheck).
*   MISRA C:2012 compliance checks.
*   Strict compiler options.
*   Unit tests for safety logic, with 100% line coverage and mutation testing.
*   Linting with Ruff and type checking with Mypy.

## Bounties

Earn bounties for contributing! See the [current bounty program](https://comma.ai/bounties)

## FAQ

*   **How do I use this?** You need a [comma 3X](https://comma.ai/shop/comma-3x).
*   **Which cars are supported?** See the [supported cars list](docs/CARS.md).
*   **Can I add support for my car?** Yes! Follow the car porting guide.
*   **Which cars can be supported?** Any car with LKAS and ACC.
*   **How does this work?** opendbc interacts with your car's systems, replacing the OEM functionality. See [this talk](https://www.youtube.com/watch?v=FL8CxUSfipM).
*   **Timeline for car support?** Car support primarily comes from the community.

### Terms

*   **port**: Specific car integration
*   **lateral control**: Steering control
*   **longitudinal control**: Acceleration/Braking control
*   **fingerprinting**: Car model identification
*   **LKAS**: Lane Keeping Assist
*   **ACC**: Adaptive Cruise Control
*   **harness**: Hardware for CAN bus connection.
*   **panda**: Hardware for CAN bus access.
*   **ECU**: Car's control modules.
*   **CAN bus**: Car's communication network.
*   **cabana**: CAN message analysis tool.
*   **DBC file**: CAN message definitions.
*   **openpilot**: ADAS system.
*   **comma**: The company.
*   **comma 3X**: Hardware to run openpilot.

### More Resources

*   [*How Do We Control The Car?*](https://www.youtube.com/watch?v=nNU6ipme878&pp=ygUoY29tbWEgY29uIDIwMjEgaG93IGRvIHdlIGNvbnRyb2wgdGhlIGNhcg%3D%3D) by [@robbederks](https://github.com/robbederks)
*   [*How to Port a Car*](https://www.youtube.com/watch?v=XxPS5TpTUnI&t=142s&pp=ygUPamFzb24gY29tbWEgY29u) by [@jyoung8607](https://github.com/jyoung8607)
*   [commaCarSegments](https://huggingface.co/datasets/commaai/commaCarSegments): CAN data from 300 car models.
*   [cabana](https://github.com/commaai/openpilot/tree/master/tools/cabana#readme)
*   [can_print_changes.py](https://github.com/commaai/openpilot/blob/master/selfdrive/debug/can_print_changes.py)
*   [longitudinal maneuvers](https://github.com/commaai/openpilot/tree/master/tools/longitudinal_maneuvers)
*   [opendbc data](https://commaai.github.io/opendbc-data/)

## Join the Team! - [comma.ai/jobs](https://comma.ai/jobs)

comma is hiring engineers to work on opendbc and [openpilot](https://github.com/commaai/openpilot).