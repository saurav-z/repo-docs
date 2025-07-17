# openpilot: Drive Smarter, Safer with Open Source AI

[openpilot](https://github.com/commaai/openpilot) is an open-source driving agent that enhances the driver assistance systems in over 300 supported car models.

## Key Features

*   **Open Source:** Built on a community-driven, open-source platform, allowing for transparency and continuous improvement.
*   **Enhanced Driver Assistance:** Upgrades existing driver assistance systems, providing features like lane keeping, adaptive cruise control, and more.
*   **Wide Car Compatibility:** Compatible with 300+ supported car models.
*   **Active Development:** Continuously updated with new features and improvements.
*   **Community Driven:** Supported by a vibrant community with active development and testing.

## Getting Started

To use openpilot, you'll need:

1.  **Supported Device:** A [comma 3/3X](https://comma.ai/shop/comma-3x).
2.  **Software:** Install the release version by using the URL `openpilot.comma.ai` during setup.
3.  **Supported Car:** One of the [275+ supported cars](docs/CARS.md).
4.  **Car Harness:** A [car harness](https://comma.ai/shop/car-harness) to connect your device to your car.

Detailed instructions are available at [comma.ai/setup](https://comma.ai/setup).

## Branches

Choose the branch that best suits your needs:

| Branch              | URL                       | Description                                                                         |
| ------------------- | ------------------------- | ------------------------------------------------------------------------------------- |
| `release3`          | openpilot.comma.ai         | The stable release branch.                                                 |
| `release3-staging`  | openpilot-test.comma.ai   | Staging branch for early access to new releases.                                       |
| `nightly`           | openpilot-nightly.comma.ai| Bleeding edge development branch, may be unstable.                                     |
| `nightly-dev`       | installer.comma.ai/commaai/nightly-dev      | Includes experimental development features.      |
| `secretgoodopenpilot` | installer.comma.ai/commaai/secretgoodopenpilot      | Preview branch from the autonomy team.      |

## Contributing

openpilot is a collaborative project. Join the community and contribute:

*   Join the [Community Discord](https://discord.comma.ai)
*   Explore the [Contributing Docs](docs/CONTRIBUTING.md)
*   Check out the [openpilot tools](tools/)
*   Access the [Code Documentation](https://docs.comma.ai)
*   Learn more on the [Community Wiki](https://github.com/commaai/openpilot/wiki)

## Safety and Testing

openpilot prioritizes safety:

*   Follows [ISO26262](https://en.wikipedia.org/wiki/ISO_26262) guidelines (see [SAFETY.md](docs/SAFETY.md)).
*   Continuous software-in-the-loop [tests](.github/workflows/selfdrive_tests.yaml)
*   Code enforcing safety model lives in panda, written in C.
*   Extensive [safety tests](https://github.com/commaai/panda/tree/master/tests/safety) for panda.
*   Hardware-in-the-loop testing with Jenkins test suite.
*   Testing closet with continuous route replay.

## License

openpilot is released under the [MIT license](LICENSE).

***

**Disclaimer:** *This is alpha-quality software for research purposes only. You are responsible for complying with local laws and regulations. NO WARRANTY EXPRESSED OR IMPLIED.*