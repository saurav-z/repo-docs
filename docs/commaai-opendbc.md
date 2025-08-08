<div align="center">
  <h1>opendbc: Your Python API for Advanced Car Control</h1>
  <p>
    <b>Unlock your car's potential!</b> Control steering, gas, brakes, and more with this powerful Python API.
  </p>

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
  <a href="https://x.com/comma_ai">
    <img src="https://img.shields.io/twitter/follow/comma_ai?style=social&label=Follow" alt="Follow @comma_ai on X">
  </a>
  <a href="https://discord.comma.ai">
    <img src="https://img.shields.io/discord/469524606043160576?label=Discord&logo=discord" alt="Join our Discord">
  </a>
</div>

---

opendbc is a Python API that gives you unprecedented control over your vehicle, enabling advanced features like automated driving and vehicle management.  Developed to support the [openpilot](https://github.com/commaai/openpilot) project, it provides a robust foundation for interacting with your car's systems.

**Key Features:**

*   🚗 **Comprehensive Car Control:** Control steering, gas, and brakes.
*   🚦 **Real-time Data Access:** Read vehicle speed, steering angle, and more.
*   🌐 **Wide Car Support:** Compatible with a growing list of vehicles (primarily post-2016 models with LKAS and ACC).  Check the [supported cars list](docs/CARS.md).
*   ⚙️ **Easy Integration:** Python API for seamless integration with your projects.
*   🛠️ **Extensible:** Easily add support for new car models and features.
*   🛡️ **Safety-Focused:** Robust safety features for reliable operation.

## Getting Started

Follow these steps to quickly get opendbc up and running:

```bash
git clone https://github.com/commaai/opendbc.git
cd opendbc

# Run a complete build and test suite
./test.sh

# Install dependencies manually
pip3 install -e .[testing,docs]
scons -j8
pytest .
lefthook run lint
```

Example programs in the [`examples/`](examples/) directory demonstrate how to read car state and control various functions, including the [`examples/joystick.py`](examples/joystick.py) script, which allows control via a joystick.

## Project Structure

*   [`opendbc/dbc/`](opendbc/dbc/): Stores [DBC](https://en.wikipedia.org/wiki/CAN_bus#DBC) files.
*   [`opendbc/can/`](opendbc/can/): Library for parsing and building CAN messages from DBC files.
*   [`opendbc/car/`](opendbc/car/): High-level library for interacting with cars using Python.
*   [`opendbc/safety/`](opendbc/safety/): Functional safety components for supported cars.

## How to Contribute: Adding Car Support

Want to add support for your car?  Follow these steps to get started:

### 1. Connect to the Car

*   Use a [comma 3X](https://comma.ai/shop/comma-3x) and a compatible car harness. If one doesn't exist, you can create your own with a "developer harness".

### 2.  Port Structure

*   A car port is located in `opendbc/car/<brand>/`.
*   Includes: `carstate.py`, `carcontroller.py`, `<brand>can.py`, `fingerprints.py`, `interface.py`, `radar_interface.py`, and `values.py`.

### 3. Reverse Engineer CAN Messages

*   Record a route with various LKAS and ACC actions.
*   Load the route in [cabana](https://github.com/commaai/openpilot/tree/master/tools/cabana) to analyze CAN data.

### 4. Tuning

*   Use the [longitudinal maneuvers](https://github.com/commaai/openpilot/tree/master/tools/longitudinal_maneuvers) tool to test and tune longitudinal control.

## Contributing

opendbc thrives on community contributions.  Get involved through GitHub and the [Discord](https://discord.comma.ai) server. Check out the `#dev-opendbc-cars` channel.

### Roadmap

*   [ ] `pip install opendbc`
*   [ ] 100% type coverage
*   [ ] 100% line coverage
*   [ ] Make car ports easier: refactors, tools, tests, and docs
*   [ ] Expose the state of all supported cars better: https://github.com/commaai/opendbc/issues/1144
*   [ ] Extend support to every car with LKAS + ACC interfaces
*   [ ] Automatic lateral and longitudinal control/tuning evaluation
*   [ ] Auto-tuning for [lateral](https://blog.comma.ai/090release/#torqued-an-auto-tuner-for-lateral-control) and longitudinal control
*   [ ] [Automatic Emergency Braking](https://en.wikipedia.org/wiki/Automated_emergency_braking_system)

## Safety Model

opendbc's safety features are designed to work with [openpilot](https://github.com/commaai/openpilot) and [panda](https://comma.ai/shop/panda). The opendbc safety firmware starts in `SAFETY_SILENT` mode and can only send messages after a safety mode is selected.

## Code Quality & Rigor

The project uses the following to ensure code quality:

*   Static code analysis with [cppcheck](https://github.com/danmar/cppcheck/)
*   [MISRA C:2012](https://misra.org.uk/) compliance checking
*   Strict compiler flags (-Wall, -Wextra, -Wstrict-prototypes, -Werror)
*   Unit tests for safety logic
*   Mutation testing on MISRA coverage
*   100% line coverage on safety unit tests
*   [Ruff linter](https://github.com/astral-sh/ruff) and [mypy](https://mypy-lang.org/) for the car interface library

## Bounties

Earn rewards for contributing:
*   $2000 - [Any car brand / platform port](https://github.com/orgs/commaai/projects/26/views/1?pane=issue&itemId=47913774)
*   $250 - [Any car model port](https://github.com/orgs/commaai/projects/26/views/1?pane=issue&itemId=47913790)
*   $300 - [Reverse Engineering a new Actuation Message](https://github.com/orgs/commaai/projects/26/views/1?pane=issue&itemId=73445563)

Higher value bounties are available for popular cars at [comma.ai/bounties](comma.ai/bounties).

## FAQ

*   **How do I use this?** Use with a [comma 3X](https://comma.ai/shop/comma-3x).
*   **Which cars are supported?** See the [supported cars list](docs/CARS.md).
*   **Can I add support for my car?** Yes, see the guide [here](https://github.com/commaai/opendbc/blob/docs/README.md#how-to-port-a-car).
*   **Which cars can be supported?** Cars with LKAS and ACC. More info [here](https://github.com/commaai/openpilot/blob/master/docs/CARS.md#dont-see-your-car-here).
*   **How does this work?** Hardware designed to replace lane keep and adaptive cruise features.  See [this talk](https://www.youtube.com/watch?v=FL8CxUSfipM).
*   **Timeline for adding car support?** Community-driven, with comma doing final safety checks.

### Terms

*   **port**:  Integration and support for a car.
*   **lateral control**: Steering control.
*   **longitudinal control**: Gas/brakes control.
*   **fingerprinting**: Identifying the car.
*   **[LKAS](https://en.wikipedia.org/wiki/Lane_departure_warning_system)**: Lane keeping assist.
*   **[ACC](https://en.wikipedia.org/wiki/Adaptive_cruise_control)**: Adaptive cruise control.
*   **[harness](https://comma.ai/shop/car-harness)**: Car-specific hardware.
*   **[panda](https://github.com/commaai/panda)**: Hardware for CAN bus access.
*   **[ECU](https://en.wikipedia.org/wiki/Electronic_control_unit)**: Car computers/control modules.
*   **[CAN bus](https://en.wikipedia.org/wiki/CAN_bus)**: Inter-ECU communication bus.
*   **[cabana](https://github.com/commaai/openpilot/tree/master/tools/cabana#readme)**: Tool for reverse engineering CAN messages.
*   **[DBC file](https://en.wikipedia.org/wiki/CAN_bus#DBC)**: Contains CAN message definitions.
*   **[openpilot](https://github.com/commaai/openpilot)**: ADAS system.
*   **[comma](https://github.com/commaai)**: The company behind opendbc.
*   **[comma 3X](https://comma.ai/shop/comma-3x)**: Hardware to run openpilot.

### More Resources

*   [*How Do We Control The Car?*](https://www.youtube.com/watch?v=nNU6ipme878&pp=ygUoY29tbWEgY29uIDIwMjEgaG93IGRvIHdlIGNvbnRyb2wgdGhlIGNhcg%3D%3D) by [@robbederks](https://github.com/robbederks)
*   [*How to Port a Car*](https://www.youtube.com/watch?v=XxPS5TpTUnI&t=142s&pp=ygUPamFzb24gY29tbWEgY29u) by [@jyoung8607](https://github.com/jyoung8607)
*   [commaCarSegments](https://huggingface.co/datasets/commaai/commaCarSegments)
*   [cabana](https://github.com/commaai/openpilot/tree/master/tools/cabana#readme)
*   [can_print_changes.py](https://github.com/commaai/openpilot/blob/master/selfdrive/debug/can_print_changes.py)
*   [longitudinal maneuvers](https://github.com/commaai/openpilot/tree/master/tools/longitudinal_maneuvers)
*   [opendbc data](https://commaai.github.io/opendbc-data/)

## Join the Team!

[comma.ai](https://comma.ai/jobs) is hiring!  Contribute to opendbc and help build the future of autonomous driving.