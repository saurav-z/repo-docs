<div align="center">
  <h1>opendbc: Your Python API for Car Control</h1>
  <p><b>Unlock your car's potential: Control steering, gas, brakes, and more with opendbc, a powerful Python API.</b></p>
  <p>
    <a href="https://docs.comma.ai">Docs</a>
    <span> · </span>
    <a href="https://github.com/commaai/openpilot/blob/master/docs/CONTRIBUTING.md">Contribute</a>
    <span> · </span>
    <a href="https://discord.comma.ai">Discord</a>
  </p>
  <a href="https://github.com/commaai/opendbc">
    <img src="https://img.shields.io/github/stars/commaai/opendbc?style=social" alt="GitHub stars">
  </a>
  <br>
  <a href="https://github.com/commaai/opendbc">
    <img src="https://img.shields.io/github/forks/commaai/opendbc?style=social" alt="GitHub forks">
  </a>
</div>

---

opendbc is a Python API designed for interacting with your car's electronic systems. It empowers you to control essential functions like steering, acceleration, and braking, while also providing real-time data on vehicle parameters. This project is primarily focused on supporting ADAS (Advanced Driver-Assistance Systems) interfaces for [openpilot](https://github.com/commaai/openpilot), but its functionality extends to reading and writing various vehicle data points.

**Key Features:**

*   **Control:** Manage steering, gas, and brakes via Python.
*   **Data Acquisition:** Read speed, steering angle, and other crucial vehicle data.
*   **Car Support:** Designed to interface with a wide range of vehicles with LKAS and ACC systems.
*   **Extensible:** Read and write various data points (EV charge status, door locks, etc.).
*   **Community Driven:** Leveraging community contributions for expanding car support.

**Get Started:**

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/commaai/opendbc.git
    cd opendbc
    ```
2.  **Run the all-in-one test script:**
    ```bash
    ./test.sh
    ```
    (This installs dependencies, builds the project, runs tests, and lints the code.)
3.  **Explore the examples:**  Check out the [`examples/`](examples/) directory for basic programs, including a joystick controller (`examples/joystick.py`).

**Project Structure:**

*   `opendbc/dbc/`:  Contains [DBC](https://en.wikipedia.org/wiki/CAN_bus#DBC) files, essential for understanding CAN bus data.
*   `opendbc/can/`:  A library for parsing and constructing CAN messages from DBC files.
*   `opendbc/car/`:  High-level Python library to interface with cars.
*   `opendbc/safety/`:  Ensures functional safety for all supported vehicles.

**How to Port a Car**

opendbc is community-driven and welcomes support for new car models.
Here is how to contribute and extend the project with new cars.

*   **Connect to the Car:** Requires a [comma 3X](https://comma.ai/shop/comma-3x) and car harness.
*   **Reverse Engineer CAN messages:** Use [cabana](https://github.com/commaai/openpilot/tree/master/tools/cabana) to analyze CAN data.
*   **Port Structure:**  Each car port includes:
    *   `carstate.py`: Parses data from the CAN stream.
    *   `carcontroller.py`: Sends control messages.
    *   `<brand>can.py`: Helpers for building CAN messages.
    *   `fingerprints.py`: Identifies car models.
    *   `interface.py`: High-level interface class.
    *   `radar_interface.py`: Parses radar data.
    *   `values.py`: Defines car-specific values.
*   **Tuning:**  Use the [longitudinal maneuvers](https://github.com/commaai/openpilot/tree/master/tools/longitudinal_maneuvers) tool.

**Contributing**

All opendbc development is coordinated on GitHub and [Discord](https://discord.comma.ai). Check out the `#dev-opendbc-cars` channel and `Vehicle Specific` section.

**Roadmap**

*   **Short Term:**
    *   `pip install opendbc`
    *   100% type and line coverage
    *   Make car ports easier: refactors, tools, tests, and docs
    *   Expose the state of all supported cars better: https://github.com/commaai/opendbc/issues/1144
*   **Longer Term:**
    *   Expand support to all LKAS + ACC equipped cars.
    *   Automatic lateral and longitudinal control/tuning evaluation.
    *   Auto-tuning for lateral and longitudinal control.
    *   Automatic Emergency Braking

**Safety Model**

When a [panda](https://comma.ai/shop/panda) powers up with [opendbc safety firmware](opendbc/safety), by default it's in `SAFETY_SILENT` mode. While in `SAFETY_SILENT` mode, the CAN buses are forced to be silent. In order to send messages, you have to select a safety mode. Some of safety modes (for example `SAFETY_ALLOUTPUT`) are disabled in release firmwares. In order to use them, compile and flash your own build.

Safety modes optionally support `controls_allowed`, which allows or blocks a subset of messages based on a customizable state in the board.

**Code Rigor**

The opendbc safety firmware is written for its use in conjunction with [openpilot](https://github.com/commaai/openpilot) and [panda](https://comma.ai/panda). The safety firmware, through its safety model, provides and enforces the
[openpilot safety](https://github.com/commaai/openpilot/blob/master/docs/SAFETY.md). Due to its critical function, it's important that the application code rigor within the `safety` folder is held to high standards.

These are the [CI regression tests](https://github.com/commaai/opendbc/actions) we have in place:
* A generic static code analysis is performed by [cppcheck](https://github.com/danmar/cppcheck/).
* In addition, [cppcheck](https://github.com/danmar/cppcheck/) has a specific addon to check for [MISRA C:2012](https://misra.org.uk/) violations. See [current coverage](opendbc/safety/tests/misra/coverage_table).
* Compiler options are relatively strict: the flags `-Wall -Wextra -Wstrict-prototypes -Werror` are enforced.
* The [safety logic](opendbc/safety) is tested and verified by [unit tests](opendbc/safety/tests) for each supported car variant.

The above tests are themselves tested by:
* a [mutation test](opendbc/safety/tests/misra/test_mutation.py) on the MISRA coverage
* 100% line coverage enforced on the safety unit tests

In addition, we run the [ruff linter](https://github.com/astral-sh/ruff) and [mypy](https://mypy-lang.org/) on the car interface library.

**Bounties**

Contribute to opendbc and earn bounties!  See details on [comma.ai/bounties](comma.ai/bounties).

**FAQ**

*   **How do I use this?**  You'll need a [comma 3X](https://comma.ai/shop/comma-3x).
*   **Which cars are supported?**  See the [supported cars list](docs/CARS.md).
*   **Can I add support for my car?**  Yes! Follow the guide [here](https://github.com/commaai/opendbc/blob/docs/README.md#how-to-port-a-car).
*   **Which cars can be supported?** Any car with LKAS and ACC. More info [here](https://github.com/commaai/openpilot/blob/master/docs/CARS.md#dont-see-your-car-here).
*   **How does this work?** Designed to replace your car's built-in lane keep and adaptive cruise features. See [this talk](https://www.youtube.com/watch?v=FL8CxUSfipM) for an in-depth explanation.
*   **Is there a timeline or roadmap for adding car support?** Most support is community-driven.

**Terms**

*   port: integration and support of a car
*   lateral control: steering control
*   longitudinal control: gas/brakes control
*   fingerprinting: automatic car identification
*   [LKAS](https://en.wikipedia.org/wiki/Lane_departure_warning_system): lane keeping assist
*   [ACC](https://en.wikipedia.org/wiki/Adaptive_cruise_control): adaptive cruise control
*   [harness](https://comma.ai/shop/car-harness): car-specific hardware
*   [panda](https://github.com/commaai/panda): CAN bus hardware
*   [ECU](https://en.wikipedia.org/wiki/Electronic_control_unit): car control module
*   [CAN bus](https://en.wikipedia.org/wiki/CAN_bus): car communication bus
*   [cabana](https://github.com/commaai/openpilot/tree/master/tools/cabana#readme): reverse engineering tool
*   [DBC file](https://en.wikipedia.org/wiki/CAN_bus#DBC): CAN message definitions
*   [openpilot](https://github.com/commaai/openpilot): ADAS system
*   [comma](https://github.com/commaai): the company behind opendbc
*   [comma 3X](https://comma.ai/shop/comma-3x): the hardware to run openpilot

**More Resources**

*   [*How Do We Control The Car?*](https://www.youtube.com/watch?v=nNU6ipme878&pp=ygUoY29tbWEgY29uIDIwMjEgaG93IGRvIHdlIGNvbnRyb2wgdGhlIGNhcg%3D%3D) by [@robbederks](https://github.com/robbederks) from COMMA_CON 2021
*   [*How to Port a Car*](https://www.youtube.com/watch?v=XxPS5TpTUnI&t=142s&pp=ygUPamFzb24gY29tbWEgY29u) by [@jyoung8607](https://github.com/jyoung8607) from COMMA_CON 2023
*   [commaCarSegments](https://huggingface.co/datasets/commaai/commaCarSegments): a massive dataset of CAN data from 300 different car models
*   [cabana](https://github.com/commaai/openpilot/tree/master/tools/cabana#readme): our tool for reverse engineering CAN messages
*   [can_print_changes.py](https://github.com/commaai/openpilot/blob/master/selfdrive/debug/can_print_changes.py): diff the whole CAN bus across two drives, such as one without any LKAS and one with LKAS
*   [longitudinal maneuvers](https://github.com/commaai/openpilot/tree/master/tools/longitudinal_maneuvers): a tool for evaluating and tuning longitudinal control
*   [opendbc data](https://commaai.github.io/opendbc-data/): a repository of longitudinal maneuver evaluations

**Join the Team!**

[comma.ai/jobs](https://comma.ai/jobs) is hiring engineers to contribute to opendbc and openpilot.

---