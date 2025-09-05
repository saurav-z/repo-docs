<div align="center" style="text-align: center;">
  <h1>opendbc: Open-Source Python API for Car Control</h1>
  <p>
    <b>Take control of your car's steering, gas, brakes, and more with opendbc!</b>
  </p>
  <br>
  <a href="https://docs.comma.ai">Docs</a>
  <span> · </span>
  <a href="https://github.com/commaai/openpilot/blob/master/docs/CONTRIBUTING.md">Contribute</a>
  <span> · </span>
  <a href="https://discord.comma.ai">Discord</a>
  <br>
  <a href="https://github.com/commaai/opendbc">
    <img src="https://img.shields.io/github/stars/commaai/opendbc?style=social" alt="GitHub stars"/>
  </a>
  <a href="https://x.com/comma_ai">
    <img src="https://img.shields.io/twitter/follow/comma_ai?style=social" alt="Twitter"/>
  </a>
  <a href="https://discord.comma.ai">
    <img src="https://img.shields.io/discord/469524606043160576?label=Discord&logo=discord" alt="Discord"/>
  </a>
</div>

---

opendbc is a powerful Python API that allows you to interface with your car's systems, enabling control over steering, acceleration, braking, and more. This open-source project empowers you to explore and expand the capabilities of modern vehicles.

**Key Features:**

*   **Control your car:** Actuate steering, gas, and brakes.
*   **Read vehicle data:** Access speed, steering angle, and other crucial information.
*   **Extensive Car Support:**  Support for many car makes and models with LKAS and ACC systems.
*   **Community Driven:**  Benefit from active community contributions.
*   **Open Source:**  Freely available under the MIT License.

**Get Started Quickly**

```bash
git clone https://github.com/commaai/opendbc.git
cd opendbc
./test.sh
```

The `test.sh` script handles installation, building, linting, and testing.

**Core Components:**

*   [`opendbc/dbc/`](opendbc/dbc/) - Contains DBC files for various car models.
*   [`opendbc/can/`](opendbc/can/) - Library for parsing and building CAN messages.
*   [`opendbc/car/`](opendbc/car/) - High-level Python library for car interaction.
*   [`opendbc/safety/`](opendbc/safety/) - Ensures functional safety for supported cars.

**How to Contribute**

opendbc thrives on community contributions!  Check out the `#dev-opendbc-cars` channel on [Discord](https://discord.comma.ai) and the `Vehicle Specific` section for discussions. Contributions are coordinated on GitHub.

**Roadmap Highlights:**

*   **[ ]**  Simplified installation: `pip install opendbc`
*   **[ ]**  100% Type and Line Coverage
*   **[ ]**  Easier Car Porting:  Improved tools and documentation.
*   **[ ]**  Enhanced car state exposure.
*   **[ ]**  Expansion to all cars with LKAS + ACC.
*   **[ ]**  Automated lateral/longitudinal control tuning.
*   **[ ]**  Support for Automated Emergency Braking (AEB).

**Safety Model**

opendbc utilizes a robust safety model, especially within the `safety` folder,  to ensure secure operation, especially when used with openpilot and a [panda](https://comma.ai/shop/panda). The safety firmware provides critical protection, with rigorous testing through:
*   Static code analysis with [cppcheck](https://github.com/danmar/cppcheck/) (including MISRA C:2012 compliance).
*   Strict compiler options ( `-Wall -Wextra -Wstrict-prototypes -Werror`).
*   Extensive unit tests for each car variant.
*   Mutation testing on MISRA coverage.
*   100% line coverage enforcement in safety unit tests.
*   Ruff linter and mypy for the car interface library

**Bounties**

Get rewarded for your contributions! Bounty programs are available for car ports and reverse engineering efforts.  See details and current bounties on the [comma.ai bounties](comma.ai/bounties) page.

**FAQ**

*   **How do I use this?**  The [comma 3X](https://comma.ai/shop/comma-3x) is the recommended hardware for running opendbc and openpilot.
*   **Which cars are supported?** See the [supported cars list](docs/CARS.md).
*   **Can I add support for my car?** Yes, the community is encouraged to add car support.  Follow the car porting guide [here](https://github.com/commaai/opendbc/blob/docs/README.md#how-to-port-a-car).
*   **Which cars can be supported?** Any car with LKAS and ACC. More info [here](https://github.com/commaai/openpilot/blob/master/docs/CARS.md#dont-see-your-car-here).
*   **How does this work?** The software and hardware are designed to replace the lane keep and adaptive cruise features already on your car. See [this talk](https://www.youtube.com/watch?v=FL8CxUSfipM) for an in-depth explanation.
*   **Is there a timeline or roadmap for adding car support?** No, most car support comes from the community.

**Terms**

*   **port**: Refers to the integration and support of a specific car
*   **lateral control**: Aka steering control
*   **longitudinal control**: Aka gas/brakes control
*   **fingerprinting**: Automatic process for identifying the car
*   **[LKAS](https://en.wikipedia.org/wiki/Lane_departure_warning_system)**: lane keeping assist
*   **[ACC](https://en.wikipedia.org/wiki/Adaptive_cruise_control)**: adaptive cruise control
*   **[harness](https://comma.ai/shop/car-harness)**: Car-specific hardware to attach to the car and intercept the ADAS messages
*   **[panda](https://github.com/commaai/panda)**: Hardware used to get on a car's CAN bus
*   **[ECU](https://en.wikipedia.org/wiki/Electronic_control_unit)**: Computers or control modules inside the car
*   **[CAN bus](https://en.wikipedia.org/wiki/CAN_bus)**: A bus that connects the ECUs in a car
*   **[cabana](https://github.com/commaai/openpilot/tree/master/tools/cabana#readme)**: A tool for reverse engineering CAN messages
*   **[DBC file](https://en.wikipedia.org/wiki/CAN_bus#DBC)**: Contains definitions for messages on a CAN bus
*   **[openpilot](https://github.com/commaai/openpilot)**: An ADAS system for cars supported by opendbc
*   **[comma](https://github.com/commaai)**: The company behind opendbc
*   **[comma 3X](https://comma.ai/shop/comma-3x)**: The hardware used to run openpilot

**More Resources**

*   [*How Do We Control The Car?*](https://www.youtube.com/watch?v=nNU6ipme878&pp=ygUoY29tbWEgY29uIDIwMjEgaG93IGRvIHdlIGNvbnRyb2wgdGhlIGNhcg%3D%3D) by [@robbederks](https://github.com/robbederks) from COMMA_CON 2021
*   [*How to Port a Car*](https://www.youtube.com/watch?v=XxPS5TpTUnI&t=142s&pp=ygUPamFzb24gY29tbWEgY29u) by [@jyoung8607](https://github.com/jyoung8607) from COMMA_CON 2023
*   [commaCarSegments](https://huggingface.co/datasets/commaai/commaCarSegments): a massive dataset of CAN data from 300 different car models
*   [cabana](https://github.com/commaai/openpilot/tree/master/tools/cabana#readme): our tool for reverse engineering CAN messages
*   [can_print_changes.py](https://github.com/commaai/openpilot/blob/master/selfdrive/debug/can_print_changes.py): diff the whole CAN bus across two drives, such as one without any LKAS and one with LKAS
*   [longitudinal maneuvers](https://github.com/commaai/openpilot/tree/master/tools/longitudinal_maneuvers): a tool for evaluating and tuning longitudinal control
*   [opendbc data](https://commaai.github.io/opendbc-data/): a repository of longitudinal maneuver evaluations

---

**Want to work with us?**  Join the team at [comma.ai/jobs](https://comma.ai/jobs) and contribute to opendbc and openpilot!  We value contributions from the community.

**[See the original repo on GitHub](https://github.com/commaai/opendbc)**