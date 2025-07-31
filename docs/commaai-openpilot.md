# openpilot: The Open Source Driver Assistance System

**Upgrade your car's driver assistance with openpilot, an open-source, community-driven autonomous driving system for over 300 supported car models.**  [Get started on GitHub](https://github.com/commaai/openpilot).

## Key Features:

*   **Enhanced Driver Assistance:**  Adds features like adaptive cruise control, lane keeping assist, and automatic lane changes to supported vehicles.
*   **Community-Driven:**  Benefit from continuous improvements and feature additions thanks to a vibrant open-source community.
*   **Wide Vehicle Support:** Compatible with over 300 car models from various manufacturers.
*   **Easy Installation:**  Simple setup using a comma 3/3X device.
*   **Regular Updates:**  Stay up-to-date with the latest features and improvements through release branches.
*   **Detailed Documentation:** Extensive documentation at [https://docs.comma.ai](https://docs.comma.ai) to guide you through setup and use.

## Getting Started:

1.  **Required Hardware:**  A comma 3/3X device, available at [comma.ai/shop](https://comma.ai/shop/comma-3x), and a compatible car harness.
2.  **Software Installation:** Install openpilot using the URL: `openpilot.comma.ai` during the setup of your comma 3/3X device.
3.  **Supported Vehicle:** Ensure your car is on the [list of supported vehicles](docs/CARS.md).
4.  **Installation Instructions:**  Refer to the detailed [setup instructions](https://comma.ai/setup) for car harness and device installation.

## Branches:

Choose the appropriate branch for your needs:

*   `release3`: Stable release branch (openpilot.comma.ai)
*   `release3-staging`: Staging branch for early access to new releases (openpilot-test.comma.ai)
*   `nightly`: Bleeding edge development branch (openpilot-nightly.comma.ai) - Expect instability.
*   `nightly-dev`: Experimental development features for some cars (installer.comma.ai/commaai/nightly-dev)
*   `secretgoodopenpilot`: Preview branch from the autonomy team (installer.comma.ai/commaai/secretgoodopenpilot)

## Contributing:

openpilot thrives on community contributions!

*   Join the discussion on the [community Discord](https://discord.comma.ai).
*   Review the [CONTRIBUTING guide](docs/CONTRIBUTING.md).
*   Explore the [openpilot tools](tools/).
*   Find code documentation at https://docs.comma.ai
*   Visit the [community wiki](https://github.com/commaai/openpilot/wiki) for user-contributed information.
*   Consider the available [bounties](https://comma.ai/bounties) or apply to [comma's open positions](https://comma.ai/jobs#open-positions) if you want to get paid to work on openpilot.

## Safety and Testing:

openpilot is committed to safety:

*   Follows [ISO26262](https://en.wikipedia.org/wiki/ISO_26262) guidelines ([SAFETY.md](docs/SAFETY.md)).
*   Runs software-in-the-loop [tests](.github/workflows/selfdrive_tests.yaml) on every commit.
*   Safety-critical code in panda written in C.
*   Panda has software-in-the-loop [safety tests](https://github.com/commaai/panda/tree/master/tests/safety).
*   Hardware-in-the-loop Jenkins test suite.
*   Panda has additional hardware-in-the-loop [tests](https://github.com/commaai/panda/blob/master/Jenkinsfile).
*   Continuous testing with multiple devices.

## License:

openpilot is released under the [MIT license](LICENSE).
*   Read the full license details in [LICENSE](LICENSE)

## Data Privacy:

By using openpilot, you agree to [our Privacy Policy](https://comma.ai/privacy).

*   You can disable data collection.
*   Data collected includes road-facing cameras, CAN, GPS, IMU, magnetometer, thermal sensors, crashes, and OS logs.
*   Driver-facing camera and microphone are optional.
*   By using openpilot, you grant comma an irrevocable, perpetual, worldwide right to use collected data.