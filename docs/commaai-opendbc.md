<div align="center">
  <h1>opendbc: Your Gateway to Car Control</h1>
  <p><b>Control and read your car's data: steering, gas, brakes, and more.</b></p>
  <p>
    <a href="https://docs.comma.ai">Docs</a>
    <span> · </span>
    <a href="https://github.com/commaai/openpilot/blob/master/docs/CONTRIBUTING.md">Contribute</a>
    <span> · </span>
    <a href="https://discord.comma.ai">Discord</a>
  </p>
  <p>
    <a href="https://github.com/commaai/opendbc"><img src="https://img.shields.io/github/stars/commaai/opendbc?style=social" alt="GitHub stars"></a>
    <a href="https://github.com/commaai/opendbc"><img src="https://img.shields.io/github/forks/commaai/opendbc?style=social" alt="GitHub forks"></a>
    <a href="https://github.com/commaai/opendbc"><img src="https://img.shields.io/github/license/commaai/opendbc?color=yellow" alt="License: MIT"></a>
    <a href="https://x.com/comma_ai"><img src="https://img.shields.io/twitter/follow/comma_ai?style=social" alt="Follow on X"></a>
    <a href="https://discord.comma.ai"><img src="https://img.shields.io/discord/469524606043160576?style=social&logo=discord" alt="Join our Discord"></a>
  </p>
</div>

---

opendbc is a powerful Python API enabling you to interface with your car's electronic systems. Leveraging advancements in automotive technology like LKAS and ACC, opendbc allows you to control steering, gas, and brakes, as well as read crucial data such as speed and steering angle. The project's primary aim is to support ADAS (Advanced Driver-Assistance Systems) interfaces for [openpilot](https://github.com/commaai/openpilot), while also providing a comprehensive toolset for vehicle management.

### Key Features

*   **Comprehensive Car Control:** Control steering, gas, and brakes.
*   **Real-time Data Access:** Read speed, steering angle, and other vital vehicle information.
*   **Extensive Car Support:**  Works with many cars, constantly expanding support. See [supported cars list](docs/CARS.md).
*   **Open Source & Community Driven:**  Contribute to car ports, add features, and shape the future of opendbc.
*   **Safety Focused:** Rigorous code testing and safety models.
*   **Easy to Use:**  Simple installation and a quick start guide.

### Quick Start

Get started quickly with these simple steps:

```bash
git clone https://github.com/commaai/opendbc.git
cd opendbc

# Run the all-in-one for dependency installation, compiling, linting, and tests
./test.sh

# For individual commands:
pip3 install -e .[testing,docs]  # Install dependencies
scons -j8                        # Build with 8 cores
pytest .                         # Run tests
lefthook run lint                # Run linter
```

Explore the [`examples/`](examples/) directory for sample programs demonstrating how to read car data and control vehicle functions, including the [`examples/joystick.py`](examples/joystick.py) example for joystick control.

### Project Structure

*   [`opendbc/dbc/`](opendbc/dbc/): Contains [DBC](https://en.wikipedia.org/wiki/CAN_bus#DBC) files, defining CAN bus messages.
*   [`opendbc/can/`](opendbc/can/): A library for parsing and building CAN messages from DBC files.
*   [`opendbc/car/`](opendbc/car/):  A high-level Python library for interfacing with cars.
*   [`opendbc/safety/`](opendbc/safety/): Contains the functional safety implementations for supported cars.

### How to Port a Car

Expand opendbc's compatibility by adding support for your car.  The process involves connecting to the car, reverse engineering CAN messages, and tuning control parameters. Detailed instructions are available for both adding new car support and enhancing existing ones.

**Steps for a New Car Port:**

1.  **Connect to the Car:** Use a [comma 3X](https://comma.ai/shop/comma-3x) and a compatible car harness.
2.  **Reverse Engineer CAN Messages:**  Use a tool like [cabana](https://github.com/commaai/openpilot/tree/master/tools/cabana) to analyze and understand CAN data.
3.  **Implement Car Port Structure:**  Create files within `opendbc/car/<brand>/` for car-specific logic.
4.  **Tune Longitudinal and Lateral Control:** Fine-tune performance using available tools and reports.

### Contributing

opendbc thrives on community contributions.  All development is coordinated via GitHub and the [Discord](https://discord.comma.ai) server. Check out the `#dev-opendbc-cars` channel.

### Roadmap

*   **Short-term:** `pip install opendbc`, 100% type and line coverage, improved car port tools and documentation.
*   **Long-term:** Expand support to every car with LKAS + ACC, enhance control and tuning.

### Safety Model

opendbc uses a safety firmware to ensure safe vehicle control. This firmware, when used with a [panda](https://comma.ai/shop/panda), provides various safety modes. Code rigor is paramount, enforced through static analysis, strict compiler settings, and rigorous testing, including unit tests and mutation testing.

### Bounties

Get rewarded for your contributions! Earn bounties for car ports and other project contributions.

*   $2000 - [Any car brand / platform port](https://github.com/orgs/commaai/projects/26/views/1?pane=issue&itemId=47913774)
*   $250 - [Any car model port](https://github.com/orgs/commaai/projects/26/views/1?pane=issue&itemId=47913790)
*   $300 - [Reverse Engineering a new Actuation Message](https://github.com/orgs/commaai/projects/26/views/1?pane=issue&itemId=73445563)

Check out [comma.ai/bounties](comma.ai/bounties) for higher-value bounties on popular cars.

### FAQ

*   **How do I use this?** A [comma 3X](https://comma.ai/shop/comma-3x) is recommended for running and developing opendbc and openpilot.
*   **Which cars are supported?** See the [supported cars list](docs/CARS.md).
*   **Can I add support for my car?** Yes!  Learn how to add your car [here](https://github.com/commaai/opendbc/blob/docs/README.md#how-to-port-a-car).
*   **Which cars can be supported?** Cars with LKAS and ACC. More information [here](https://github.com/commaai/openpilot/blob/master/docs/CARS.md#dont-see-your-car-here).
*   **How does this work?**  opendbc works by intercepting ADAS messages. See [this talk](https://www.youtube.com/watch?v=FL8CxUSfipM) for a deep dive.
*   **Is there a timeline or roadmap for adding car support?** Support comes from the community; comma validates.

### Terms

*   **port**: specific car integration and support.
*   **lateral control**: steering control.
*   **longitudinal control**: gas/brake control.
*   **fingerprinting**: identifying the car automatically.
*   **[LKAS](https://en.wikipedia.org/wiki/Lane_departure_warning_system)**: lane keeping assist.
*   **[ACC](https://en.wikipedia.org/wiki/Adaptive_cruise_control)**: adaptive cruise control.
*   **[harness](https://comma.ai/shop/car-harness)**: car-specific hardware for ADAS message interception.
*   **[panda](https://github.com/commaai/panda)**: hardware for CAN bus access.
*   **[ECU](https://en.wikipedia.org/wiki/Electronic_control_unit)**: in-car computers/control modules.
*   **[CAN bus](https://en.wikipedia.org/wiki/CAN_bus)**: inter-ECU communication bus.
*   **[cabana](https://github.com/commaai/openpilot/tree/master/tools/cabana#readme)**: tool for reverse engineering CAN messages.
*   **[DBC file](https://en.wikipedia.org/wiki/CAN_bus#DBC)**: CAN bus message definitions.
*   **[openpilot](https://github.com/commaai/openpilot)**: ADAS system using opendbc.
*   **[comma](https://github.com/commaai)**: company behind opendbc.
*   **[comma 3X](https://comma.ai/shop/comma-3x)**: hardware for running openpilot.

### More Resources

*   [*How Do We Control The Car?*](https://www.youtube.com/watch?v=nNU6ipme878&pp=ygUoY29tbWEgY29uIDIwMjEgaG93IGRvIHdlIGNvbnRyb2wgdGhlIGNhcg%3D%3D) by [@robbederks](https://github.com/robbederks) from COMMA_CON 2021
*   [*How to Port a Car*](https://www.youtube.com/watch?v=XxPS5TpTUnI&t=142s&pp=ygUPamFzb24gY29tbWEgY29u) by [@jyoung8607](https://github.com/jyoung8607) from COMMA_CON 2023
*   [commaCarSegments](https://huggingface.co/datasets/commaai/commaCarSegments):  CAN data from 300 car models.
*   [cabana](https://github.com/commaai/openpilot/tree/master/tools/cabana#readme): Tool to reverse engineer CAN messages
*   [can_print_changes.py](https://github.com/commaai/openpilot/blob/master/selfdrive/debug/can_print_changes.py): Diff the whole CAN bus across drives.
*   [longitudinal maneuvers](https://github.com/commaai/openpilot/tree/master/tools/longitudinal_maneuvers): Tool for evaluating and tuning longitudinal control
*   [opendbc data](https://commaai.github.io/opendbc-data/): Repository of longitudinal maneuver evaluations

## Join the Team!

Explore career opportunities at [comma.ai/jobs](https://comma.ai/jobs). We are hiring engineers for opendbc and [openpilot](https://github.com/commaai/openpilot).