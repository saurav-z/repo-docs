<div align="center">
  <h1>opendbc: Your Python API for Vehicle Control</h1>
  <p><b>Unlock your car's potential with opendbc: a powerful Python API to control steering, gas, brakes, and more.</b></p>

  <p>
    <a href="https://docs.comma.ai">Docs</a>
    <span> · </span>
    <a href="https://github.com/commaai/openpilot/blob/master/docs/CONTRIBUTING.md">Contribute</a>
    <span> · </span>
    <a href="https://discord.comma.ai">Discord</a>
  </p>

  <p>
    <a href="https://github.com/commaai/opendbc">
      <img src="https://img.shields.io/github/stars/commaai/opendbc?style=social" alt="GitHub stars">
    </a>
    <a href="https://x.com/comma_ai">
      <img src="https://img.shields.io/twitter/follow/comma_ai?style=social" alt="Follow @comma_ai on X">
    </a>
    <a href="https://discord.comma.ai">
      <img src="https://img.shields.io/discord/469524606043160576?style=social" alt="Join our Discord">
    </a>
  </p>

</div>

---

opendbc provides a comprehensive Python API for interacting with your car's internal systems. Since 2016, many cars have implemented electronically-actuated steering, gas, and brakes.  opendbc gives you the power to control these systems.  It's the foundation for the openpilot project but designed for broader vehicle management applications.  The project's documentation, including the [supported cars list](docs/CARS.md), provides all the information you need to get started.

## Key Features

*   **Control & Read Vehicle Systems**: Access and control steering, gas, brakes, and other vehicle data.
*   **DBC File Integration**: Utilize DBC files for CAN message parsing and building.
*   **Car-Specific Libraries**: High-level libraries for easy interfacing with various car models.
*   **Safety Model**: Built-in safety features to ensure secure vehicle control.
*   **Community Driven**: Benefit from a thriving community and contributions.
*   **Extensive Documentation**:  Comprehensive documentation, including the [supported cars list](docs/CARS.md) and examples.

## Quickstart

```bash
git clone https://github.com/commaai/opendbc.git
cd opendbc

# Install dependencies, build, run tests, and lint the code (recommended)
./test.sh
```

Alternatively, run these commands individually:

```bash
pip3 install -e .[testing,docs]  # install dependencies
scons -j8                        # build with 8 cores
pytest .                         # run the tests
lefthook run lint                # run the linter
```

Explore the example programs in the [`examples/`](examples/) directory to read and control your car's systems.  The [`examples/joystick.py`](examples/joystick.py) example lets you control a car with a joystick.

## Project Structure

*   [`opendbc/dbc/`](opendbc/dbc/): Stores [DBC](https://en.wikipedia.org/wiki/CAN_bus#DBC) files for various cars.
*   [`opendbc/can/`](opendbc/can/): Contains a library for parsing and generating CAN messages from DBC files.
*   [`opendbc/car/`](opendbc/car/): Provides a high-level library for interacting with cars via Python.
*   [`opendbc/safety/`](opendbc/safety/):  Implements the functional safety aspects for supported cars.

## Contributing & Car Porting

### How to Port a Car

Expand your car's compatibility by following our guide to adding support for new vehicles.  Whether you're adding lateral control or integrating radar parsing, we guide you through every step.

1.  **Connect to the Car:** Use a comma 3X with a car harness.
2.  **Harness**: Find a compatible harness or create one with a developer harness.
3.  **Structure of a port**:  Car ports reside in `opendbc/car/<brand>/`.
    *   `carstate.py`: Parses CAN data.
    *   `carcontroller.py`: Outputs CAN messages.
    *   `<brand>can.py`: Helpers for building CAN messages.
    *   `fingerprints.py`: Identifies ECU firmware.
    *   `interface.py`: High-level car interface.
    *   `radar_interface.py`: Parses radar data.
    *   `values.py`: Defines supported cars.
4.  **Reverse Engineer CAN Messages**: Use [cabana](https://github.com/commaai/openpilot/tree/master/tools/cabana) to analyze captured CAN data.
5.  **Tuning**: Evaluate and fine-tune your car's performance using the [longitudinal maneuvers](https://github.com/commaai/openpilot/tree/master/tools/longitudinal_maneuvers) tool.

### Contributing

Contribute to opendbc's development on GitHub and the [Discord server](https://discord.comma.ai).  Check out the `#dev-opendbc-cars` channel and the `Vehicle Specific` section.

## Roadmap

*   **Short Term**:
    *   `pip install opendbc`
    *   100% type coverage.
    *   100% line coverage.
    *   Improve car port development with refactors, tools, tests, and documentation.
    *   Improve state of supported cars better: https://github.com/commaai/opendbc/issues/1144.
*   **Longer Term**:
    *   Extend support to all cars with LKAS + ACC.
    *   Automated lateral and longitudinal control and tuning evaluation.
    *   Auto-tuning for [lateral](https://blog.comma.ai/090release/#torqued-an-auto-tuner-for-lateral-control) and longitudinal control.
    *   [Automatic Emergency Braking](https://en.wikipedia.org/wiki/Automated_emergency_braking_system)

## Safety Model

opendbc's safety firmware is used with openpilot and the panda hardware. The default `SAFETY_SILENT` mode silences the CAN buses. Select a safety mode to transmit messages. Some safety modes are disabled in release firmwares. You can compile your own build to enable them.  Safety modes optionally support `controls_allowed`.

## Code Rigor

The opendbc safety firmware is designed for openpilot and panda, which enforces [openpilot safety](https://github.com/commaai/openpilot/blob/master/docs/SAFETY.md). Rigorous testing is critical:

*   Static code analysis with [cppcheck](https://github.com/danmar/cppcheck/), including [MISRA C:2012](https://misra.org.uk/) checks. See [current coverage](opendbc/safety/tests/misra/coverage_table).
*   Strict compiler flags: `-Wall -Wextra -Wstrict-prototypes -Werror`.
*   Unit tests for each car variant in [safety logic](opendbc/safety).
*   100% line coverage enforcement on safety unit tests.
*   Mutation tests on the MISRA coverage.
*   Ruff linter and [mypy](https://mypy-lang.org/) for the car interface library.

### Bounties

Earn bounties for contributing:

*   $2000 - [Any car brand / platform port](https://github.com/orgs/commaai/projects/26/views/1?pane=issue&itemId=47913774)
*   $250 - [Any car model port](https://github.com/orgs/commaai/projects/26/views/1?pane=issue&itemId=47913790)
*   $300 - [Reverse Engineering a new Actuation Message](https://github.com/orgs/commaai/projects/26/views/1?pane=issue&itemId=73445563)

Visit [comma.ai/bounties](comma.ai/bounties) for higher-value bounties on popular car models.

## FAQ

**Q: How do I use this?**
A: Use a [comma 3X](https://comma.ai/shop/comma-3x).

**Q: Which cars are supported?**
A: See the [supported cars list](docs/CARS.md).

**Q: Can I add support for my car?**
A: Yes, most car support comes from the community.  Read the guide [here](https://github.com/commaai/opendbc/blob/docs/README.md#how-to-port-a-car).

**Q: Which cars can be supported?**
A: Any car with LKAS and ACC. More info [here](https://github.com/commaai/openpilot/blob/master/docs/CARS.md#dont-see-your-car-here).

**Q: How does this work?**
A: We designed hardware to replace your car's built-in lane keep and adaptive cruise features. See [this talk](https://www.youtube.com/watch?v=FL8CxUSfipM).

**Q: Is there a timeline or roadmap for adding car support?**
A: No, it's community-driven with final comma validation.

### Terms

*   **port**: Integrating and supporting a car.
*   **lateral control**: Steering control.
*   **longitudinal control**: Gas/brakes control.
*   **fingerprinting**: Automatic car identification.
*   **[LKAS](https://en.wikipedia.org/wiki/Lane_departure_warning_system)**: Lane keeping assist.
*   **[ACC](https://en.wikipedia.org/wiki/Adaptive_cruise_control)**: Adaptive cruise control.
*   **[harness](https://comma.ai/shop/car-harness)**: Car-specific hardware.
*   **[panda](https://github.com/commaai/panda)**: Hardware for CAN bus access.
*   **[ECU](https://en.wikipedia.org/wiki/Electronic_control_unit)**: Car computers.
*   **[CAN bus](https://en.wikipedia.org/wiki/CAN_bus)**: Inter-ECU communication bus.
*   **[cabana](https://github.com/commaai/openpilot/tree/master/tools/cabana#readme)**: CAN message reverse engineering tool.
*   **[DBC file](https://en.wikipedia.org/wiki/CAN_bus#DBC)**: Defines CAN messages.
*   **[openpilot](https://github.com/commaai/openpilot)**: ADAS system.
*   **[comma](https://github.com/commaai)**: The company behind opendbc.
*   **[comma 3X](https://comma.ai/shop/comma-3x)**: Hardware for running openpilot.

### More Resources

*   [*How Do We Control The Car?*](https://www.youtube.com/watch?v=nNU6ipme878&pp=ygUoY29tbWEgY29uIDIwMjEgaG93IGRvIHdlIGNvbnRyb2wgdGhlIGNhcg%3D%3D) by [@robbederks](https://github.com/robbederks) from COMMA_CON 2021
*   [*How to Port a Car*](https://www.youtube.com/watch?v=XxPS5TpTUnI&t=142s&pp=ygUPamFzb24gY29tbWEgY29u) by [@jyoung8607](https://github.com/jyoung8607) from COMMA_CON 2023
*   [commaCarSegments](https://huggingface.co/datasets/commaai/commaCarSegments): Massive dataset of CAN data from 300 different car models
*   [cabana](https://github.com/commaai/openpilot/tree/master/tools/cabana#readme): Our tool for reverse engineering CAN messages
*   [can_print_changes.py](https://github.com/commaai/openpilot/blob/master/selfdrive/debug/can_print_changes.py): Diff the whole CAN bus across two drives
*   [longitudinal maneuvers](https://github.com/commaai/openpilot/tree/master/tools/longitudinal_maneuvers): Tool for evaluating and tuning longitudinal control
*   [opendbc data](https://commaai.github.io/opendbc-data/): Repository of longitudinal maneuver evaluations

## Join the Team -- [comma.ai/jobs](https://comma.ai/jobs)

Comma is hiring engineers to work on opendbc and [openpilot](https://github.com/commaai/openpilot).  We welcome contributions!