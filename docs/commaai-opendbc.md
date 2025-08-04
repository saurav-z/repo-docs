# opendbc: Your Python API for Automotive Control and Data

**opendbc empowers you to control and read data from your car, opening the door to advanced automotive applications.**  [Explore the opendbc repository on GitHub](https://github.com/commaai/opendbc).

## Key Features

*   **Control:** Take command of steering, gas, and brakes.
*   **Read Data:** Access vital information like speed and steering angle.
*   **Extensive Car Support:**  Designed to support a wide range of vehicles with LKAS and ACC.
*   **Open Source:** Contribute to a community-driven project.
*   **Python API:**  Leverage a user-friendly Python interface.
*   **Safety Focused:**  Built with rigorous code rigor for safety-critical applications.

## Quick Start

Get started by cloning the repository and running the test script:

```bash
git clone https://github.com/commaai/opendbc.git
cd opendbc
./test.sh
```

## Project Structure

*   [`opendbc/dbc/`](opendbc/dbc/):  DBC file repository (CAN bus definitions).
*   [`opendbc/can/`](opendbc/can/):  Library for parsing and building CAN messages.
*   [`opendbc/car/`](opendbc/car/):  High-level Python library for car interfacing.
*   [`opendbc/safety/`](opendbc/safety/):  Safety-critical code for supported cars.

## How to Contribute

All development is coordinated on GitHub and [Discord](https://discord.comma.ai). Check out the `#dev-opendbc-cars` channel and `Vehicle Specific` section.

## How to Port a Car

This guide covers everything from adding support to a new car all the way to improving existing cars (e.g. adding longitudinal control or radar parsing). If similar cars to yours are already compatible, most of this work is likely already done for you.

### Connect to the Car

The first step is to get connected to the car with a comma 3X and a car harness.
The car harness gets you connected to two different CAN buses and splits one of those buses to send our own actuation messages.

If you're lucky, a harness compatible with your car will already be designed and sold on comma.ai/shop.
If you're not so lucky, start with a "developer harness" from comma.ai/shop and crimp on whatever connector you need.

### Structure of a port

Depending on the brand, most of this basic structure will already be in place.

The entirety of a car port lives in `opendbc/car/<brand>/`:
* `carstate.py`: parses out the relevant information from the CAN stream using the car's DBC file
* `carcontroller.py`: outputs CAN messages to control the car
* `<brand>can.py`: thin Python helpers around the DBC file to build CAN messages
* `fingerprints.py`: database of ECU firmware versions for identifying car models
* `interface.py`: high level class for interfacing with the car
* `radar_interface.py`: parses out the radar
* `values.py`: enumerates the brand's supported cars

### Reverse Engineer CAN messages

Start off by recording a route with lots of interesting events: enable LKAS and ACC, turn the steering wheel both extremes, etc. Then, load up that route in [cabana](https://github.com/commaai/openpilot/tree/master/tools/cabana).

### Tuning

#### Longitudinal

Use the [longitudinal maneuvers](https://github.com/commaai/openpilot/tree/master/tools/longitudinal_maneuvers) report to evaluate your car's longitudinal control and tune it.

## Roadmap

*   [ ]  `pip install opendbc`
*   [ ]  100% type coverage
*   [ ]  100% line coverage
*   [ ]  Make car ports easier: refactors, tools, tests, and docs
*   [ ]  Expose the state of all supported cars better: https://github.com/commaai/opendbc/issues/1144

## Safety Model

The opendbc safety firmware, in conjunction with openpilot and panda, ensures safe operation.

## Code Rigor

The opendbc safety firmware code undergoes stringent testing, including:

*   Static code analysis (cppcheck)
*   MISRA C:2012 compliance checks
*   Strict compiler flags
*   Unit tests with 100% line coverage
*   Mutation testing

## Bounties

Earn bounties for contributing!

*   $2000 - [Any car brand / platform port](https://github.com/orgs/commaai/projects/26/views/1?pane=issue&itemId=47913774)
*   $250 - [Any car model port](https://github.com/orgs/commaai/projects/26/views/1?pane=issue&itemId=47913790)
*   $300 - [Reverse Engineering a new Actuation Message](https://github.com/orgs/commaai/projects/26/views/1?pane=issue&itemId=73445563)

See [comma.ai/bounties](https://comma.ai/bounties) for more information.

## FAQ

**How do I use this?**  A [comma 3X](https://comma.ai/shop/comma-3x) is recommended for developing with opendbc and openpilot.

**Which cars are supported?**  See the [supported cars list](docs/CARS.md).

**Can I add support for my car?** Yes, the community drives most car support. Read the guide [here](https://github.com/commaai/opendbc/blob/docs/README.md#how-to-port-a-car).

**Which cars can be supported?**  Any car with LKAS and ACC. More info [here](https://github.com/commaai/openpilot/blob/master/docs/CARS.md#dont-see-your-car-here).

**How does this work?**  Hardware replaces your car's lane keep and adaptive cruise features. See [this talk](https://www.youtube.com/watch?v=FL8CxUSfipM).

**Is there a timeline or roadmap for adding car support?**  Community-driven with comma.ai validating and finalizing.

## Terms

*   **port**: Car integration
*   **lateral control**: Steering control
*   **longitudinal control**: Gas/brakes control
*   **fingerprinting**: Car model identification
*   **LKAS**: Lane Keeping Assist
*   **ACC**: Adaptive Cruise Control
*   **harness**: Car-specific hardware
*   **panda**: CAN bus interface hardware
*   **ECU**: Electronic Control Unit
*   **CAN bus**: Car communication bus
*   **cabana**: CAN message reverse engineering tool
*   **DBC file**: CAN bus message definitions
*   **openpilot**: ADAS system
*   **comma**: The company behind opendbc
*   **comma 3X**: Hardware for running openpilot

## More Resources

*   [*How Do We Control The Car?*](https://www.youtube.com/watch?v=nNU6ipme878&pp=ygUoY29tbWEgY29uIDIwMjEgaG93IGRvIHdlIGNvbnRyb2wgdGhlIGNhcg%3D%3D)
*   [*How to Port a Car*](https://www.youtube.com/watch?v=XxPS5TpTUnI&t=142s&pp=ygUPamFzb24gY29tbWEgY29u)
*   [commaCarSegments](https://huggingface.co/datasets/commaai/commaCarSegments)
*   [cabana](https://github.com/commaai/openpilot/tree/master/tools/cabana#readme)
*   [can_print_changes.py](https://github.com/commaai/openpilot/blob/master/selfdrive/debug/can_print_changes.py)
*   [longitudinal maneuvers](https://github.com/commaai/openpilot/tree/master/tools/longitudinal_maneuvers)
*   [opendbc data](https://commaai.github.io/opendbc-data/)

## Join the Team - [comma.ai/jobs](https://comma.ai/jobs)

comma.ai is hiring engineers to work on opendbc and [openpilot](https://github.com/commaai/openpilot).