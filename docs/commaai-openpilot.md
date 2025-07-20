# Openpilot: Drive Smarter, Safer, and Freely with Open Source Autonomous Driving

[Openpilot](https://github.com/commaai/openpilot) is a cutting-edge open-source autonomous driving system, enhancing driver assistance in over 300 supported vehicles.

## Key Features:

*   **Enhanced Driver Assistance:** Upgrade your car's existing driver assistance systems with advanced features.
*   **Open Source & Community Driven:** Benefit from a collaborative community and contribute to the project's development.
*   **Wide Vehicle Compatibility:** Supports 300+ car models, with the list constantly growing.
*   **Continuous Development:** Benefit from new features and improvements through various branches like `release3`, `release3-staging`, `nightly`, `nightly-dev`, and `secretgoodopenpilot`.
*   **Comprehensive Testing:** Rigorous testing, including software and hardware-in-the-loop tests, ensures safety and reliability.
*   **Data-Driven Improvement:** The system utilizes driving data to refine models and enhance performance.

## Getting Started with Openpilot

To experience the power of openpilot, you'll need:

1.  **Supported Device:** A [comma 3/3X](https://comma.ai/shop/comma-3x).
2.  **Software:** Install the release version using the URL `openpilot.comma.ai`.
3.  **Supported Car:** One of [the 275+ supported cars](docs/CARS.md).
4.  **Car Harness:** A [car harness](https://comma.ai/shop/car-harness) to connect your comma 3/3X to your car.

Detailed installation instructions are available at [comma.ai/setup](https://comma.ai/setup).

## Contributing

Openpilot thrives on community contributions. Join us!

*   [Contribute to the project](https://github.com/commaai/openpilot/blob/master/docs/CONTRIBUTING.md)
*   Join the [community Discord](https://discord.comma.ai)
*   Explore the [openpilot tools](tools/)
*   Access comprehensive documentation at [https://docs.comma.ai](https://docs.comma.ai)
*   Get the latest community updates on the [community wiki](https://github.com/commaai/openpilot/wiki)

## Safety and Testing

Openpilot prioritizes safety and rigorous testing:

*   Adheres to [ISO26262](https://en.wikipedia.org/wiki/ISO_26262) guidelines ([SAFETY.md](docs/SAFETY.md)).
*   Features software-in-the-loop tests that run on every commit.
*   Employs safety-critical code written in C for the panda system.
*   Utilizes software and hardware-in-the-loop tests in the panda system.
*   Uses a hardware-in-the-loop Jenkins test suite.
*   Continuously tests in a testing closet with multiple comma devices.

## Licensing and Data Usage

*   Openpilot is released under the [MIT License](LICENSE).
*   By default, openpilot uploads driving data to comma's servers to improve the system. Users can access their data via [comma connect](https://connect.comma.ai/).  Users are free to disable data collection if they wish.
*   By using openpilot, you agree to [Comma.ai's Privacy Policy](https://comma.ai/privacy).

**THIS IS ALPHA QUALITY SOFTWARE FOR RESEARCH PURPOSES ONLY. THIS IS NOT A PRODUCT.
YOU ARE RESPONSIBLE FOR COMPLYING WITH LOCAL LAWS AND REGULATIONS.
NO WARRANTY EXPRESSED OR IMPLIED.**