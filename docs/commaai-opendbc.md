# opendbc: Your Python API to Your Car - Unlock Advanced Vehicle Control

**opendbc** empowers you to control and read data from your car, providing a Python API for advanced vehicle access.  Explore the power of opendbc and the future of vehicle technology.

[View the opendbc GitHub Repository](https://github.com/commaai/opendbc)

**Key Features:**

*   **Control & Read Data:** Control steering, gas, brakes, and access data like speed and steering angle.
*   **Open Source & Community Driven:**  Benefit from a collaborative environment with active community contributions.
*   **ADAS Support:** Primarily focuses on supporting ADAS interfaces for openpilot and future vehicle management applications.
*   **Extensive Car Support:**  Designed to support a wide range of vehicles with LKAS and ACC features.
*   **Safety Focused:**  Prioritizes code rigor and safety in its core safety firmware.
*   **Reverse Engineering Tools:** Includes tools like Cabana for CAN message analysis.
*   **Bounties:**  Earn bounties for contributing to car port development.

---

opendbc provides a powerful and flexible framework for interacting with your car's systems, with a primary focus on ADAS features for openpilot. Whether you're interested in controlling your vehicle or extracting valuable data, opendbc offers a comprehensive solution.

**Getting Started:**

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/commaai/opendbc.git
    cd opendbc
    ```
2.  **Run the `test.sh` script:**
    ```bash
    ./test.sh
    ```
    This script handles dependency installation, building, linting, and running tests.

3.  **Explore Examples:** Examine the example programs in the `examples/` directory, such as `examples/joystick.py`, to control your car with a joystick.

**Project Structure:**

*   `opendbc/dbc/`:  Contains DBC (Database CAN) files.
*   `opendbc/can/`:  Provides a library for parsing and building CAN messages from DBC files.
*   `opendbc/car/`:  Offers a high-level Python library for interacting with cars.
*   `opendbc/safety/`:  Includes the safety-critical code and firmware, crucial for vehicle control.

**How to Port a Car:**

The `opendbc` project welcomes contributions from the community!  The guide to porting new cars is available in the [README](docs/README.md). Follow these steps to add support for your vehicle:

1.  **Connect to the Car:** Connect to the car using a comma 3X and car harness.
2.  **Reverse Engineer CAN Messages:** Use Cabana to analyze CAN data and identify the relevant messages.
3.  **Implement a Port:**  Create the necessary files within the `opendbc/car/<brand>/` directory.
4.  **Tuning:** Use tools to refine the car's performance.

**Contributing:**

All development is coordinated on GitHub and [Discord](https://discord.comma.ai).

**Roadmap:**

*   [ ] `pip install opendbc`
*   [ ] 100% type coverage
*   [ ] 100% line coverage
*   [ ] Make car ports easier: refactors, tools, tests, and docs
*   [ ] Expose the state of all supported cars better: https://github.com/commaai/opendbc/issues/1144

**Safety Model & Code Rigor:**

opendbc's safety firmware is written with high standards, including static code analysis, MISRA C:2012 compliance checks, strict compiler flags, and rigorous unit testing.

**Bounties:**

Earn rewards for contributing to the project!  Bounties are offered for adding new car ports, reverse engineering messages, and more. See [comma.ai/bounties](comma.ai/bounties) for details.

**FAQ:**

*   **How do I use this?** Requires a [comma 3X](https://comma.ai/shop/comma-3x)
*   **Which cars are supported?** See the [supported cars list](docs/CARS.md).
*   **Can I add support for my car?**  Yes! Read the car porting guide.
*   **Which cars can be supported?** Any car with LKAS and ACC. More info [here](https://github.com/commaai/openpilot/blob/master/docs/CARS.md#dont-see-your-car-here).
*   **How does this work?** Designed hardware to replace your car's built-in lane keep and adaptive cruise features. See [this talk](https://www.youtube.com/watch?v=FL8CxUSfipM) for an in-depth explanation.
*   **Is there a timeline or roadmap for adding car support?** The community primarily drives the car porting process.

**More Resources:**

*   [How Do We Control The Car?](https://www.youtube.com/watch?v=nNU6ipme878&pp=ygUoY29tbWEgY29uIDIwMjEgaG93IGRvIHdlIGNvbnRyb2wgdGhlIGNhcg%3D%3D) by [@robbederks](https://github.com/robbederks) from COMMA_CON 2021
*   [How to Port a Car](https://www.youtube.com/watch?v=XxPS5TpTUnI&t=142s&pp=ygUPamFzb24gY29tbWEgY29u) by [@jyoung8607](https://github.com/jyoung8607) from COMMA_CON 2023
*   [commaCarSegments](https://huggingface.co/datasets/commaai/commaCarSegments): a massive dataset of CAN data from 300 different car models
*   [cabana](https://github.com/commaai/openpilot/tree/master/tools/cabana#readme): our tool for reverse engineering CAN messages
*   [can_print_changes.py](https://github.com/commaai/openpilot/blob/master/selfdrive/debug/can_print_changes.py): diff the whole CAN bus across two drives, such as one without any LKAS and one with LKAS
*   [longitudinal maneuvers](https://github.com/commaai/openpilot/tree/master/tools/longitudinal_maneuvers): a tool for evaluating and tuning longitudinal control
*   [opendbc data](https://commaai.github.io/opendbc-data/): a repository of longitudinal maneuver evaluations

**Join the team!**

[comma.ai/jobs](https://comma.ai/jobs) is hiring engineers to work on opendbc and [openpilot](https://github.com/commaai/openpilot).