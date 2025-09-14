<div align="center" style="text-align: center;">
  <h1>opendbc: Python API for Your Car</h1>
  <p>
    <b>Unlock your car's potential with opendbc, a Python API that gives you control over steering, gas, brakes, and more.</b> <br>
    Read and write data to your vehicle's systems for advanced control and data analysis.
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

## Key Features

*   🚗 **Control Your Car:** Take command of steering, gas, and brakes.
*   📡 **Read Vehicle Data:** Access speed, steering angle, and other critical information.
*   ⚙️ **Open Source & Community Driven:** Contribute to a project supported by a vibrant community.
*   🛠️ **Expandable & Flexible:** Designed to support a wide range of vehicles and features, including EV charge status, door locks, and more.
*   🐍 **Python API:** Easy to use and integrate with your existing Python projects.

[opendbc](https://github.com/commaai/opendbc) provides a powerful Python API for interacting with your car's systems. This allows you to control steering, gas, and brakes, as well as read vital information like speed and steering angle. Whether you're building an advanced driver-assistance system (ADAS) or simply exploring your vehicle's capabilities, opendbc gives you the tools you need.

---

## Quick Start

Get started with opendbc by cloning the repository and running the test script:

```bash
git clone https://github.com/commaai/opendbc.git
cd opendbc

# Run the all-in-one script for dependencies, building, linting, and testing:
./test.sh
```

For individual commands, see the original README.

Explore the `examples/` directory for simple programs demonstrating car state reading and control, including a joystick-controlled driving example (`examples/joystick.py`).

## Project Structure

*   `opendbc/dbc/`: Contains [DBC](https://en.wikipedia.org/wiki/CAN_bus#DBC_(CAN_Database_Files)) files, defining CAN message structures.
*   `opendbc/can/`: A library for parsing and building CAN messages from DBC files.
*   `opendbc/car/`: A high-level Python library for interfacing with car systems.
*   `opendbc/safety/`: Implements safety features for supported cars.

## How to Port a Car

Adding support for a new car involves connecting to the car's CAN buses, reverse engineering CAN messages, and creating the necessary car-specific modules.

The process includes:

1.  **Connect to the Car:** Use a comma 3X and a car harness.
2.  **Structure of a port:** Utilize files within `opendbc/car/<brand>/`.
3.  **Reverse Engineer CAN messages:** Analyze CAN data using tools like cabana.
4.  **Tuning:** Optimize your car's performance by using tools like the longitudinal maneuvers report.

Complete car ports include:

*   lateral control
*   longitudinal control
*   tuning
*   radar parsing (if equipped)
*   fingerprinting

For more detailed instructions, please refer to the documentation.

## Contributing

opendbc thrives on community contributions. Join us on GitHub and [Discord](https://discord.comma.ai) to get involved.

## Roadmap

The project roadmap includes:

*   `pip install opendbc`
*   100% type and line coverage
*   Improved car port creation
*   Extending car support and more.

See the full roadmap in the original README.

## Safety Model

The opendbc safety firmware provides critical safety features. It is written in conjunction with [openpilot](https://github.com/commaai/openpilot) and [panda](https://github.com/commaai/panda).

## Code Rigor

The safety firmware uses:

*   Static code analysis
*   Compiler options
*   Unit tests
*   Mutation testing
*   Linters

## Bounties

We offer bounties for contributing to opendbc. Learn more at [comma.ai/bounties](comma.ai/bounties).

## FAQ

**How do I use this?** A [comma 3X](https://comma.ai/shop/comma-3x) is custom-designed to be the best way to run and develop opendbc and openpilot.
**Which cars are supported?** See the [supported cars list](docs/CARS.md).
**Can I add support for my car?** Yes. Read the guide [here](https://github.com/commaai/opendbc/blob/docs/README.md#how-to-port-a-car).
**Which cars can be supported?** Any car with LKAS and ACC. More info [here](https://github.com/commaai/openpilot/blob/master/docs/CARS.md#dont-see-your-car-here).
**How does this work?**  See [this talk](https://www.youtube.com/watch?v=FL8CxUSfipM) for an in-depth explanation.
**Is there a timeline or roadmap for adding car support?** No, most car support comes from the community, with comma doing final safety and quality validation.

## Terms

See the original README for a detailed glossary.

## More Resources

See the original README for more resources.

## Come work with us -- [comma.ai/jobs](https://comma.ai/jobs)

Join the team at comma.ai and help build the future of automotive technology!