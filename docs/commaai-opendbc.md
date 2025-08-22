<div align="center" style="text-align: center;">
  <h1>opendbc: Open Source Car Interface API</h1>
  <p><b>opendbc unlocks the power to control and read your car's systems with a Python API, opening doors to advanced vehicle control and management.</b></p>

  <p>
    <a href="https://docs.comma.ai">Docs</a>
    <span> · </span>
    <a href="https://github.com/commaai/openpilot/blob/master/docs/CONTRIBUTING.md">Contribute</a>
    <span> · </span>
    <a href="https://discord.comma.ai">Discord</a>
  </p>

  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
  [![X Follow](https://img.shields.io/twitter/follow/comma_ai)](https://x.com/comma_ai)
  [![Discord](https://img.shields.io/discord/469524606043160576)](https://discord.comma.ai)
</div>

---

## Key Features

*   **Control Your Car:** Access and manipulate essential vehicle functions like steering, gas, and brakes.
*   **Real-Time Data:** Read crucial data points such as speed and steering angle.
*   **Broad Compatibility:** Designed to support a wide range of vehicles, especially those with LKAS and ACC.
*   **Open Source & Community Driven:** Benefit from a collaborative ecosystem and contribute to vehicle control innovation.
*   **Safety-Focused:** Rigorous code standards and safety models ensure reliable and secure operation.
*   **Comprehensive Documentation:** Complete guides and resources for developers, contributors, and enthusiasts.

---

## Getting Started

opendbc is a Python API designed to provide low-level access to your vehicle. Clone the repository from [GitHub](https://github.com/commaai/opendbc) and then install the dependencies.

```bash
git clone https://github.com/commaai/opendbc.git
cd opendbc

# Install dependencies, build, and run tests with one command
./test.sh
```

Explore the `examples/` directory for practical code samples and demonstrations. You can also use `examples/joystick.py` to control the car with a joystick.

## Project Structure

*   **`opendbc/dbc/`**: Stores [DBC](https://en.wikipedia.org/wiki/CAN_bus#DBC) files for various vehicles.
*   **`opendbc/can/`**: Contains a library for parsing and constructing CAN messages.
*   **`opendbc/car/`**: Offers a high-level Python API for interacting with car systems.
*   **`opendbc/safety/`**: Provides functional safety features for supported vehicles.

## Contributing

Contribute to opendbc's development on GitHub and [Discord](https://discord.comma.ai). Join the `#dev-opendbc-cars` channel and the `Vehicle Specific` section to connect with the community.

### Roadmap

**Short Term**
*   [ ] `pip install opendbc`
*   [ ] 100% type coverage
*   [ ] 100% line coverage
*   [ ] Make car ports easier: refactors, tools, tests, and docs
*   [ ] Expose the state of all supported cars better: https://github.com/commaai/opendbc/issues/1144

**Longer Term**
*   [ ] Extend support to every car with LKAS + ACC interfaces
*   [ ] Automatic lateral and longitudinal control/tuning evaluation
*   [ ] Auto-tuning for [lateral](https://blog.comma.ai/090release/#torqued-an-auto-tuner-for-lateral-control) and longitudinal control
*   [ ] [Automatic Emergency Braking](https://en.wikipedia.org/wiki/Automated_emergency_braking_system)

## Safety Model

The opendbc safety firmware, used with the [panda](https://comma.ai/shop/panda), enforces safety protocols. When the panda powers up, it will be in `SAFETY_SILENT` mode.  You have to select a safety mode to begin sending messages.  Code rigor is prioritized within the `safety` folder, with [CI regression tests](https://github.com/commaai/opendbc/actions) to ensure reliability. These include static code analysis, MISRA C:2012 checks, strict compiler options, and comprehensive unit tests.

## Bounties

Contribute and earn! Explore bounties for various tasks:

*   $2000 - [Any car brand / platform port](https://github.com/orgs/commaai/projects/26/views/1?pane=issue&itemId=47913774)
*   $250 - [Any car model port](https://github.com/orgs/commaai/projects/26/views/1?pane=issue&itemId=47913790)
*   $300 - [Reverse Engineering a new Actuation Message](https://github.com/orgs/commaai/projects/26/views/1?pane=issue&itemId=73445563)

For more valuable bounties, visit [comma.ai/bounties](comma.ai/bounties).

## Frequently Asked Questions (FAQ)

*   **How do I use this?** A [comma 3X](https://comma.ai/shop/comma-3x) is recommended for opendbc and openpilot development.

*   **Which cars are supported?** See the [supported cars list](docs/CARS.md).

*   **Can I add support for my car?** Yes, community contributions are welcome. See the guide [here](https://github.com/commaai/opendbc/blob/docs/README.md#how-to-port-a-car).

*   **Which cars can be supported?** Any car with LKAS and ACC. More info [here](https://github.com/commaai/openpilot/blob/master/docs/CARS.md#dont-see-your-car-here).

*   **How does this work?** It replaces your car's lane keep and adaptive cruise features.  See [this talk](https://www.youtube.com/watch?v=FL8CxUSfipM) for more details.

*   **Is there a timeline or roadmap for adding car support?**  Community contributions drive car support.

### Definitions

*   **port**: The integration and support of a specific car.
*   **lateral control**: Steering control.
*   **longitudinal control**: Gas/brakes control.
*   **fingerprinting**: Automatic process for identifying the car.
*   **[LKAS](https://en.wikipedia.org/wiki/Lane_departure_warning_system)**: Lane keeping assist.
*   **[ACC](https://en.wikipedia.org/wiki/Adaptive_cruise_control)**: Adaptive cruise control.
*   **[harness](https://comma.ai/shop/car-harness)**: Car-specific hardware for ADAS message interception.
*   **[panda](https://github.com/commaai/panda)**: Hardware for CAN bus access.
*   **[ECU](https://en.wikipedia.org/wiki/Electronic_control_unit)**: Car's computer modules.
*   **[CAN bus](https://en.wikipedia.org/wiki/CAN_bus)**: Connects ECUs in a car.
*   **[cabana](https://github.com/commaai/openpilot/tree/master/tools/cabana#readme)**: Tool for reverse engineering CAN messages.
*   **[DBC file](https://en.wikipedia.org/wiki/CAN_bus#DBC)**: Defines CAN bus message contents.
*   **[openpilot](https://github.com/commaai/openpilot)**: ADAS system supported by opendbc.
*   **[comma](https://github.com/commaai)**: The company behind opendbc.
*   **[comma 3X](https://comma.ai/shop/comma-3x)**: Hardware used to run openpilot.

### Further Resources

*   [*How Do We Control The Car?*](https://www.youtube.com/watch?v=nNU6ipme878&pp=ygUoY29tbWEgY29uIDIwMjEgaG93IGRvIHdlIGNvbnRyb2wgdGhlIGNhcg%3D%3D) by [@robbederks](https://github.com/robbederks) from COMMA_CON 2021
*   [*How to Port a Car*](https://www.youtube.com/watch?v=XxPS5TpTUnI&t=142s&pp=ygUPamFzb24gY29tbWEgY29u) by [@jyoung8607](https://github.com/jyoung8607) from COMMA_CON 2023
*   [commaCarSegments](https://huggingface.co/datasets/commaai/commaCarSegments): Dataset of CAN data from various car models
*   [cabana](https://github.com/commaai/openpilot/tree/master/tools/cabana#readme): Reverse engineering tool for CAN messages.
*   [can_print_changes.py](https://github.com/commaai/openpilot/blob/master/selfdrive/debug/can_print_changes.py): Tool for diffing CAN bus data.
*   [longitudinal maneuvers](https://github.com/commaai/openpilot/tree/master/tools/longitudinal_maneuvers): Tool for evaluating and tuning longitudinal control.
*   [opendbc data](https://commaai.github.io/opendbc-data/): Repository of longitudinal maneuver evaluations

## Join the Team at [comma.ai/jobs](https://comma.ai/jobs)

Join the comma.ai team of engineers to work on opendbc and [openpilot](https://github.com/commaai/openpilot).