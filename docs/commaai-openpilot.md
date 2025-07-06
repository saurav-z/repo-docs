# openpilot: Open Source Driver Assistance for a Smarter Ride

**openpilot is an open-source, community-driven driver-assistance system, currently enhancing the driving experience in over 300 supported car models.** Learn more and contribute on the [original GitHub repository](https://github.com/commaai/openpilot).

## Key Features

*   **Enhanced Driver Assistance:** Upgrade your car's existing driver-assistance features with cutting-edge technology.
*   **Wide Vehicle Support:** Compatible with over 300 car models, and the list is continuously growing.
*   **Community-Driven:** Benefit from a vibrant community of developers and users, constantly improving the system.
*   **Open Source & Customizable:** Modify and adapt the software to your specific needs and contribute to its evolution.
*   **Regular Updates:** Stay up-to-date with the latest features and improvements through continuous development and testing.

## Getting Started

To use openpilot, you'll need:

1.  **Supported Device:** A comma 3/3X, available at [comma.ai/shop](https://comma.ai/shop/comma-3x).
2.  **Software:** Install openpilot on your comma 3/3X using the URL: `openpilot.comma.ai`.
3.  **Supported Car:** Ensure your car is listed among the [supported models](docs/CARS.md).
4.  **Car Harness:** A [car harness](https://comma.ai/shop/car-harness) to connect your device to your car.

## Installation

Detailed installation instructions are available at [comma.ai/setup](https://comma.ai/setup).

## Branches

Select a branch to install, with the following options:

*   `release3`: `openpilot.comma.ai` - Openpilot's stable release.
*   `release3-staging`: `openpilot-test.comma.ai` - Staging branch for early access to releases.
*   `nightly`: `openpilot-nightly.comma.ai` - Bleeding-edge development branch (unstable).
*   `nightly-dev`: `installer.comma.ai/commaai/nightly-dev` - Nightly with experimental features for some cars.
*   `secretgoodopenpilot`: `installer.comma.ai/commaai/secretgoodopenpilot` - Preview branch with new driving models.

## Contributing

openpilot is developed by [comma](https://comma.ai/) and the community.

*   Join the [community Discord](https://discord.comma.ai)
*   Review the [contributing docs](docs/CONTRIBUTING.md)
*   Explore the [openpilot tools](tools/)
*   Find code documentation at https://docs.comma.ai
*   Learn more on the [community wiki](https://github.com/commaai/openpilot/wiki)

[comma is hiring](https://comma.ai/jobs#open-positions) and offers bounties for external contributors.

## Safety and Testing

*   openpilot adheres to [ISO26262](https://en.wikipedia.org/wiki/ISO_26262) guidelines (see [SAFETY.md](docs/SAFETY.md)).
*   Automated software-in-the-loop tests run on every commit ([.github/workflows/selfdrive_tests.yaml](https://github.com/commaai/openpilot/actions/workflows/selfdrive_tests.yaml)).
*   Safety model code is in panda written in C (see [code rigor](https://github.com/commaai/panda#code-rigor)).
*   Panda has software-in-the-loop [safety tests](https://github.com/commaai/panda/tree/master/tests/safety).
*   Hardware-in-the-loop Jenkins test suite.
*   Panda has additional hardware-in-the-loop [tests](https://github.com/commaai/panda/blob/master/Jenkinsfile).
*   Testing is performed continuously on multiple comma devices.

## License

openpilot is released under the MIT license.

## User Data and comma Account

openpilot uploads driving data to comma's servers, accessible via [comma connect](https://connect.comma.ai/). You can disable data collection.

openpilot logs data like camera feeds, CAN, GPS, and other sensor data. Driver-facing camera and microphone are only logged with explicit opt-in.

By using openpilot, you agree to [our Privacy Policy](https://comma.ai/privacy).