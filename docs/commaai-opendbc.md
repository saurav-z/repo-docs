<div align="center" style="text-align: center;">
  <h1>opendbc: Your Python API for Car Control</h1>
  <p>
    <b>Take control of your car's steering, gas, and brakes with opendbc!</b>
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

opendbc is a powerful Python API enabling developers to interact with and control various car systems, developed by [comma.ai](https://github.com/commaai/opendbc). It provides granular control over critical functions like steering, acceleration, and braking, opening doors for advanced automotive applications.

**Key Features:**

*   **Car Control:** Directly manipulate steering, gas, and brake systems.
*   **Data Acquisition:** Read real-time data including speed, steering angle, and more.
*   **DBC File Integration:** Utilizes DBC files for parsing and building CAN messages.
*   **Extensive Car Support:** Designed to support a wide range of vehicles with LKAS and ACC.
*   **Safety-Focused:** Includes a robust safety model for reliable and secure operation.
*   **Open Source & Community Driven:** Benefit from community contributions and a collaborative development environment.
*   **Complete Car Integration:** Interfacing with car using Python
*   **Reverse Engineering CAN messages:** Reverse engineer car's CAN messages

### Getting Started

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/commaai/opendbc.git
    cd opendbc
    ```

2.  **Run the test script:**
    ```bash
    ./test.sh
    ```

    This script installs dependencies, builds the project, runs tests, and lints the code.

3.  **Explore Examples:**  Find example programs in the [`examples/`](examples/) directory. The [`examples/joystick.py`](examples/joystick.py) file lets you control a car with a joystick.

### Project Structure

*   `opendbc/dbc/`:  Stores DBC files for different car models.
*   `opendbc/can/`:  Provides a library for parsing and constructing CAN messages.
*   `opendbc/car/`:  Offers a high-level Python API for interacting with vehicles.
*   `opendbc/safety/`:  Implements the functional safety mechanisms.

### How to Port a Car

This section explains how to add support for a new car or improve existing ones.
A car port will control the steering on a car, it will have all of: lateral control, longitudinal control, good tuning for both lateral and longitudinal, radar parsing (if equipped), fuzzy fingerprinting, and more. The new car support docs will clearly communicate each car's support level.

1.  **Connect to the Car:** Use a comma 3X with a compatible car harness.
2.  **Structure of a Port:**  A car port typically resides in `opendbc/car/<brand>/` and includes:
    *   `carstate.py`: Parses data from the CAN stream.
    *   `carcontroller.py`: Sends CAN messages to control the car.
    *   `<brand>can.py`: Helper functions for building CAN messages.
    *   `fingerprints.py`: Identifies car models based on ECU firmware.
    *   `interface.py`: The high-level interface class.
    *   `radar_interface.py`: Handles radar data parsing.
    *   `values.py`: Lists supported car models.
3.  **Reverse Engineer CAN messages:** Use cabana to analyze CAN data.
4.  **Tuning:** Use the [longitudinal maneuvers](https://github.com/commaai/openpilot/tree/master/tools/longitudinal_maneuvers) report to evaluate and tune the car's longitudinal control.

### Contributing

All opendbc development is coordinated on GitHub and [Discord](https://discord.comma.ai).

*   **Roadmap:**
    *   [ ] `pip install opendbc`
    *   [ ] 100% type coverage
    *   [ ] 100% line coverage
    *   [ ] Make car ports easier: refactors, tools, tests, and docs
    *   [ ] Expose the state of all supported cars better: https://github.com/commaai/opendbc/issues/1144
    *   [ ] Extend support to every car with LKAS + ACC interfaces
    *   [ ] Automatic lateral and longitudinal control/tuning evaluation
    *   [ ] Auto-tuning for [lateral](https://blog.comma.ai/090release/#torqued-an-auto-tuner-for-lateral-control) and longitudinal control
    *   [ ] [Automatic Emergency Braking](https://en.wikipedia.org/wiki/Automated_emergency_braking_system)

### Safety Model

The `opendbc/safety` folder contains safety-critical firmware for use with [openpilot](https://github.com/commaai/openpilot) and [panda](https://github.com/commaai/panda). It operates in different safety modes, including a silent mode and modes that enforce safety checks.

*   **Code Rigor:** High standards are maintained, with static analysis, MISRA C:2012 compliance checks, strict compiler flags, unit tests, mutation tests, and code coverage enforcement.

### Bounties

Earn bounties for your contributions:

*   $2000 - [Any car brand / platform port](https://github.com/orgs/commaai/projects/26/views/1?pane=issue&itemId=47913774)
*   $250 - [Any car model port](https://github.com/orgs/commaai/projects/26/views/1?pane=issue&itemId=47913790)
*   $300 - [Reverse Engineering a new Actuation Message](https://github.com/orgs/commaai/projects/26/views/1?pane=issue&itemId=73445563)

Additional bounties are available for popular cars at [comma.ai/bounties](comma.ai/bounties).

### FAQ

*   **How do I use this?** Use a [comma 3X](https://comma.ai/shop/comma-3x).
*   **Which cars are supported?** See the [supported cars list](docs/CARS.md).
*   **Can I add support for my car?** Yes. See the guide [here](https://github.com/commaai/opendbc/blob/docs/README.md#how-to-port-a-car).
*   **Which cars can be supported?** Any car with LKAS and ACC. More info [here](https://github.com/commaai/openpilot/blob/master/docs/CARS.md#dont-see-your-car-here).
*   **How does this work?** opendbc replaces your car's lane keep and adaptive cruise features. See [this talk](https://www.youtube.com/watch?v=FL8CxUSfipM) for more details.
*   **Is there a timeline or roadmap for adding car support?** No, community-driven.

### Terms

*   ... (Terms defined as per original README)

### More resources

*   ... (Resources as per original README)

### Get Involved: Join the comma.ai Team!  [comma.ai/jobs](https://comma.ai/jobs)

comma is actively hiring engineers to contribute to opendbc and [openpilot](https://github.com/commaai/openpilot).  We encourage contributions from the community!