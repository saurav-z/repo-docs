# openpilot: Open Source Driver Assistance for Your Car

**Upgrade your driving experience with openpilot, an open-source, community-driven autonomous driving system that enhances the driver assistance features in hundreds of supported vehicles.** Learn more and explore the project on the original repository: [https://github.com/commaai/openpilot](https://github.com/commaai/openpilot).

## Key Features

*   **Enhanced Driver Assistance:** Provides features like adaptive cruise control, lane keeping assist, and automatic lane centering.
*   **Wide Vehicle Compatibility:** Supports over 300+ car models.
*   **Open Source & Community Driven:** Benefit from continuous improvements and contributions from a vibrant community of developers and users.
*   **Easy Installation:** Quick setup with the comma 3/3X hardware.
*   **Data-Driven Improvement:** Uses data to train and improve the autonomous driving models, benefiting all users.

## How to Get Started

To use openpilot in your car, you'll need:

1.  **Supported Device:** A comma 3/3X, available at [comma.ai/shop](https://comma.ai/shop/comma-3x).
2.  **Software:** Install the release version using the URL `openpilot.comma.ai`.
3.  **Supported Car:** Check the list of supported vehicles:  [docs/CARS.md](docs/CARS.md).
4.  **Car Harness:** A [car harness](https://comma.ai/shop/car-harness) to connect your device to your car.

Detailed instructions for installation are available at [comma.ai/setup](https://comma.ai/setup).

## Branches

Choose the branch that best suits your needs:

*   **`release3`:** Stable release branch (`openpilot.comma.ai`)
*   **`release3-staging`:** Staging branch for early access to new releases (`openpilot-test.comma.ai`)
*   **`nightly`:** Bleeding-edge development branch (unstable) (`openpilot-nightly.comma.ai`)
*   **`nightly-dev`:** Development branch with experimental features for some cars (`installer.comma.ai/commaai/nightly-dev`)
*   **`secretgoodopenpilot`:** Preview branch with early driving models

## Contributing

Contribute to openpilot and help improve autonomous driving technology.

*   Join the [community Discord](https://discord.comma.ai)
*   Read the [contributing docs](docs/CONTRIBUTING.md)
*   Explore the [openpilot tools](tools/)
*   Find code documentation at https://docs.comma.ai
*   Learn more on the [community wiki](https://github.com/commaai/openpilot/wiki)

**Interested in getting paid to work on openpilot?** [comma is hiring](https://comma.ai/jobs#open-positions) and offers many [bounties](https://comma.ai/bounties) for external contributors.

## Safety and Testing

openpilot prioritizes safety through rigorous testing and adherence to safety standards.

*   Observes [ISO26262](https://en.wikipedia.org/wiki/ISO_26262) guidelines, see [SAFETY.md](docs/SAFETY.md)
*   Software-in-the-loop [tests](.github/workflows/selfdrive_tests.yaml) run on every commit.
*   Safety model code is in panda and written in C, see [code rigor](https://github.com/commaai/panda#code-rigor)
*   panda has software-in-the-loop [safety tests](https://github.com/commaai/panda/tree/master/tests/safety).
*   Internal hardware-in-the-loop Jenkins test suite.
*   panda has hardware-in-the-loop [tests](https://github.com/commaai/panda/blob/master/Jenkinsfile).
*   Continuous testing with multiple comma devices replaying routes.

## License

[MIT License](LICENSE)

**Disclaimer:** This is alpha-quality software for research purposes only. It is not a product. You are responsible for complying with local laws and regulations. NO WARRANTY EXPRESSED OR IMPLIED.

## User Data and comma Account

By using openpilot, you agree that driving data is uploaded to our servers and can be accessed via [comma connect](https://connect.comma.ai/). We use your data to improve the models. You can disable data collection. By using openpilot, you agree to [our Privacy Policy](https://comma.ai/privacy).