<div align="center">
  <h1>opendbc: Open Source Car Interface Library</h1>
  <p><b>Unlock the full potential of your car with opendbc, a powerful Python API for accessing and controlling vehicle systems.</b></p>
  <br>
  <a href="https://github.com/commaai/opendbc">
      <img src="https://img.shields.io/github/stars/commaai/opendbc?style=social" alt="GitHub Stars">
  </a>
</div>

---

opendbc is a comprehensive Python library that provides an interface to your car's internal systems, allowing you to read and write data such as steering, gas, brakes, and more. Whether you're a seasoned developer, a car enthusiast, or just curious, opendbc provides the tools you need to explore and interact with your vehicle. This README is your primary guide to the project, and we encourage you to read the [supported cars list](docs/CARS.md) to see if your car is supported.

## Key Features

*   **Control & Monitor:** Access and control critical car functions like steering, acceleration, and braking.
*   **Data Extraction:** Read real-time data including speed, steering angle, and more.
*   **Extensive Car Support:** Designed to support a wide range of vehicles with LKAS and ACC interfaces.
*   **Open Source & Community Driven:** Benefit from a collaborative ecosystem and contribute to the project.
*   **Safety Focused:** Rigorous code rigor and testing to ensure safety.

## Getting Started

Clone the repository and run the test script:

```bash
git clone https://github.com/commaai/opendbc.git
cd opendbc
./test.sh
```

This script handles dependency installation, compilation, linting, and testing.  You can also run the commands individually:

```bash
pip3 install -e .[testing,docs]  # install dependencies
scons -j8                        # build with 8 cores
pytest .                         # run the tests
lefthook run lint                # run the linter
```

Explore the [`examples/`](examples/) directory for sample programs, like [`examples/joystick.py`](examples/joystick.py), which allows you to control a car with a joystick.

## Project Structure

*   [`opendbc/dbc/`](opendbc/dbc/) - Contains DBC files for various car models.
*   [`opendbc/can/`](opendbc/can/) - A library for parsing and building CAN messages.
*   [`opendbc/car/`](opendbc/car/) - A high-level Python library for interacting with cars.
*   [`opendbc/safety/`](opendbc/safety/) - Functional safety components for supported cars.

## How to Port a Car

Adding support for a new car involves understanding the car's CAN bus communication.  If your car is similar to an already supported vehicle, much of the groundwork may already be done for you.

### Essential Steps

1.  **Connect to the Car:** Use a [comma 3X](https://comma.ai/shop/comma-3x) and a compatible car harness.
2.  **CAN Message Reverse Engineering:** Utilize [cabana](https://github.com/commaai/openpilot/tree/master/tools/cabana) to analyze CAN data.
3.  **Develop the port:** Create the following files, specific to the brand of car:
    *   `carstate.py`
    *   `carcontroller.py`
    *   `<brand>can.py`
    *   `fingerprints.py`
    *   `interface.py`
    *   `radar_interface.py`
    *   `values.py`
4.  **Tuning:** Refine control using tools like the [longitudinal maneuvers](https://github.com/commaai/openpilot/tree/master/tools/longitudinal_maneuvers) report.

## Contributing

Join the opendbc community on GitHub and [Discord](https://discord.comma.ai). Check out the `#dev-opendbc-cars` channel and the `Vehicle Specific` section.

### Roadmap

*   [ ] `pip install opendbc`
*   [ ] 100% type coverage
*   [ ] 100% line coverage
*   [ ] Make car ports easier: refactors, tools, tests, and docs
*   [ ] Expose the state of all supported cars better: https://github.com/commaai/opendbc/issues/1144
*   [ ] Extend support to every car with LKAS + ACC interfaces
*   [ ] Automatic lateral and longitudinal control/tuning evaluation
*   [ ] Auto-tuning for lateral and longitudinal control
*   [ ] Automatic Emergency Braking

## Safety Model

The opendbc safety firmware, used with [openpilot](https://github.com/commaai/openpilot) and [panda](https://comma.ai/shop/panda), enforces the [openpilot safety](https://github.com/commaai/openpilot/blob/master/docs/SAFETY.md). This is accomplished using strict code rigor, including static analysis with [cppcheck](https://github.com/danmar/cppcheck/), [MISRA C:2012](https://misra.org.uk/) checks, and comprehensive unit tests with 100% line coverage.

## Bounties

Contribute to opendbc and earn rewards!

*   $2000 - [Any car brand / platform port](https://github.com/orgs/commaai/projects/26/views/1?pane=issue&itemId=47913774)
*   $250 - [Any car model port](https://github.com/orgs/commaai/projects/26/views/1?pane=issue&itemId=47913790)
*   $300 - [Reverse Engineering a new Actuation Message](https://github.com/orgs/commaai/projects/26/views/1?pane=issue&itemId=73445563)

Additional bounties are available for popular car models at [comma.ai/bounties](comma.ai/bounties).

## Frequently Asked Questions

*   **How do I use this?** A [comma 3X](https://comma.ai/shop/comma-3x) is custom-designed to be the best way to run and develop opendbc and openpilot.
*   **Which cars are supported?** See the [supported cars list](docs/CARS.md).
*   **Can I add support for my car?** Yes, most car support comes from the community. Read the guide [here](https://github.com/commaai/opendbc/blob/docs/README.md#how-to-port-a-car).
*   **Which cars can be supported?** Any car with LKAS and ACC. More info [here](https://github.com/commaai/openpilot/blob/master/docs/CARS.md#dont-see-your-car-here).
*   **How does this work?** In short, we designed hardware to replace your car's built-in lane keep and adaptive cruise features. See [this talk](https://www.youtube.com/watch?v=FL8CxUSfipM) for an in-depth explanation.
*   **Is there a timeline or roadmap for adding car support?** No, most car support comes from the community, with comma doing final safety and quality validation. The more complete the community car port is and the more popular the car is, the more likely we are to pick it up as the next one to validate.

### Terms

*   **port**: refers to the integration and support of a specific car
*   **lateral control**: aka steering control
*   **longitudinal control**: aka gas/brakes control
*   **fingerprinting**: automatic process for identifying the car
*   **[LKAS](https://en.wikipedia.org/wiki/Lane_departure_warning_system)**: lane keeping assist
*   **[ACC](https://en.wikipedia.org/wiki/Adaptive_cruise_control)**: adaptive cruise control
*   **[harness](https://comma.ai/shop/car-harness)**: car-specific hardware to attach to the car and intercept the ADAS messages
*   **[panda](https://github.com/commaai/panda)**: hardware used to get on a car's CAN bus
*   **[ECU](https://en.wikipedia.org/wiki/Electronic_control_unit)**: computers or control modules inside the car
*   **[CAN bus](https://en.wikipedia.org/wiki/CAN_bus)**: a bus that connects the ECUs in a car
*   **[cabana](https://github.com/commaai/openpilot/tree/master/tools/cabana#readme)**: our tool for reverse engineering CAN messages
*   **[DBC file](https://en.wikipedia.org/wiki/CAN_bus#DBC)**: contains definitions for messages on a CAN bus
*   **[openpilot](https://github.com/commaai/openpilot)**: an ADAS system for cars supported by opendbc
*   **[comma](https://github.com/commaai)**: the company behind opendbc
*   **[comma 3X](https://comma.ai/shop/comma-3x)**: the hardware used to run openpilot

### More Resources

*   [*How Do We Control The Car?*](https://www.youtube.com/watch?v=nNU6ipme878&pp=ygUoY29tbWEgY29uIDIwMjEgaG93IGRvIHdlIGNvbnRyb2wgdGhlIGNhcg%3D%3D) by [@robbederks](https://github.com/robbederks) from COMMA_CON 2021
*   [*How to Port a Car*](https://www.youtube.com/watch?v=XxPS5TpTUnI&t=142s&pp=ygUPamFzb24gY29tbWEgY29u) by [@jyoung8607](https://github.com/jyoung8607) from COMMA_CON 2023
*   [commaCarSegments](https://huggingface.co/datasets/commaai/commaCarSegments): a massive dataset of CAN data from 300 different car models
*   [cabana](https://github.com/commaai/openpilot/tree/master/tools/cabana#readme): our tool for reverse engineering CAN messages
*   [can_print_changes.py](https://github.com/commaai/openpilot/blob/master/selfdrive/debug/can_print_changes.py): diff the whole CAN bus across two drives, such as one without any LKAS and one with LKAS
*   [longitudinal maneuvers](https://github.com/commaai/openpilot/tree/master/tools/longitudinal_maneuvers): a tool for evaluating and tuning longitudinal control
*   [opendbc data](https://commaai.github.io/opendbc-data/): a repository of longitudinal maneuver evaluations

## Join the Team

comma is hiring engineers! Explore job openings at [comma.ai/jobs](https://comma.ai/jobs).