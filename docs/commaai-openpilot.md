# openpilot: The Open Source Driver Assistance System

**Openpilot is a cutting-edge open-source software project that upgrades the driver assistance systems in over 300 supported car models.** Explore a new dimension of driving with advanced features and continuous improvement, all thanks to the power of open-source technology! Learn more about openpilot on the original repository: [https://github.com/commaai/openpilot](https://github.com/commaai/openpilot)

## Key Features

*   **Automated Driver Assistance:** Enjoy features like adaptive cruise control, lane keeping assist, and automatic lane changes.
*   **Broad Vehicle Support:**  Enhance your drive with compatibility for 300+ car models (check the [CARS.md](docs/CARS.md) file).
*   **Community-Driven Development:** Benefit from continuous improvements and a thriving community of developers and enthusiasts.
*   **Open Source:**  Contribute, customize, and learn from the freely available source code under the MIT license.
*   **Data-Driven Improvement:**  Driving data is uploaded to improve openpilot performance (with the option to disable data collection).
*   **Easy Installation:** Start with a [comma 3/3X](https://comma.ai/shop/comma-3x) and install openpilot with a single command: `bash <(curl -fsSL openpilot.comma.ai)`
*   **Multiple Release Branches:** Choose the right openpilot experience with `release3` for stable builds, `release3-staging` for early access, `nightly` for the cutting edge, and `nightly-dev` for experimental features.

## Getting Started

To use openpilot, you'll need:

1.  **Supported Device:** A comma 3/3X, available at [comma.ai/shop](https://comma.ai/shop/comma-3x).
2.  **Software:** Install the release version by using the URL `openpilot.comma.ai`.
3.  **Supported Car:**  One of the [275+ supported cars](docs/CARS.md).
4.  **Car Harness:**  A [car harness](https://comma.ai/shop/car-harness) to connect your comma 3/3X to your car.

For detailed setup instructions, visit [comma.ai/setup](https://comma.ai/setup).

## Development

*   **Join the Community:** [Discord](https://discord.comma.ai)
*   **Contribute:** Learn how to contribute at [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md)
*   **Explore the Code:** Check out the [openpilot tools](tools/)
*   **Documentation:** Find comprehensive documentation at [https://docs.comma.ai](https://docs.comma.ai)
*   **Community Wiki:** Information about running openpilot lives on the [community wiki](https://github.com/commaai/openpilot/wiki)
*   **Job Opportunities & Bounties:** Consider working for comma and earning bounties - [comma.ai/jobs#open-positions](https://comma.ai/jobs#open-positions) & [comma.ai/bounties](https://comma.ai/bounties)

## Safety and Testing

*   Follows [ISO26262](https://en.wikipedia.org/wiki/ISO_26262) guidelines ([SAFETY.md](docs/SAFETY.md) for more details).
*   Includes software-in-the-loop tests ([.github/workflows/selfdrive_tests.yaml](.github/workflows/selfdrive_tests.yaml))
*   Safety model code in panda (C) with code rigor details: [code rigor](https://github.com/commaai/panda#code-rigor)
*   panda has software-in-the-loop safety tests: [safety tests](https://github.com/commaai/panda/tree/master/tests/safety)
*   Internal hardware-in-the-loop Jenkins test suite.
*   panda has additional hardware-in-the-loop tests: [panda tests](https://github.com/commaai/panda/blob/master/Jenkinsfile).
*   Continuous testing with comma devices replaying routes in a testing closet.

## License and Data Usage

*   **License:** MIT ([LICENSE](LICENSE)).
*   **Data:**  openpilot uploads driving data to improve performance. You can access your data through [comma connect](https://connect.comma.ai/).  Data collection can be disabled. By using openpilot, you agree to the [Privacy Policy](https://comma.ai/privacy).

**Disclaimer:** This is alpha-quality software for research purposes only.  You are responsible for complying with local laws and regulations.  NO WARRANTY EXPRESSED OR IMPLIED.