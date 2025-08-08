# openpilot: Open Source Driver Assistance System

**openpilot is a cutting-edge, open-source driver-assistance system that enhances the capabilities of your vehicle, offering advanced features like adaptive cruise control and lane keeping assistance for a safer and more convenient driving experience.** ([Original Repo](https://github.com/commaai/openpilot))

## Key Features

*   **Open Source:**  Fully transparent and community-driven, fostering innovation and allowing users to understand and modify the system.
*   **Supported Vehicles:** Enhances driver assistance in 300+ supported car models; check compatibility at [docs/CARS.md](docs/CARS.md).
*   **Advanced Driver Assistance:** Provides features such as adaptive cruise control, lane keeping assist, and automatic lane centering.
*   **Community Driven:** Benefit from a vibrant community through our [Discord server](https://discord.comma.ai) and actively contribute to openpilot's development.
*   **Regular Updates:**  Stay up-to-date with the latest features and improvements via various branches including release, staging, and nightly builds.
*   **Safety Focused:** Adheres to [ISO26262](https://en.wikipedia.org/wiki/ISO_26262) guidelines, with rigorous testing and safety measures, detailed in [SAFETY.md](docs/SAFETY.md).
*   **Data-Driven Improvement:**  Uses data from users to continuously improve and refine the system's performance, with options for user data access and management via [comma connect](https://connect.comma.ai/).

## Getting Started

To start using openpilot, you'll need:

1.  **Compatible Device:** A comma 3/3X, available at [comma.ai/shop/comma-3x](https://comma.ai/shop/comma-3x).
2.  **Software:** Install the release version using the URL `openpilot.comma.ai`.
3.  **Supported Car:** Ensure your car is listed in the [supported cars list](docs/CARS.md).
4.  **Car Harness:**  A [car harness](https://comma.ai/shop/car-harness) to connect your device to your car.

Detailed installation instructions are available at [comma.ai/setup](https://comma.ai/setup).

## Branches

Choose the branch that fits your needs:

| Branch           | URL                                    | Description                                                                         |
|------------------|----------------------------------------|-------------------------------------------------------------------------------------|
| `release3`       | openpilot.comma.ai                      | Stable release branch.                                                             |
| `release3-staging` | openpilot-test.comma.ai                | Staging branch for early access to new releases.                                     |
| `nightly`          | openpilot-nightly.comma.ai             | Bleeding-edge development branch; expect instability.                              |
| `nightly-dev`      | installer.comma.ai/commaai/nightly-dev | Nightly build with experimental development features for some cars.                  |
| `secretgoodopenpilot` | installer.comma.ai/commaai/secretgoodopenpilot | Preview branch with new driving models for the autonomy team. |

## Contributing

We welcome contributions from the community!

*   Join the [community Discord](https://discord.comma.ai).
*   Review the [contributing documentation](docs/CONTRIBUTING.md).
*   Explore the [openpilot tools](tools/).
*   Access code documentation at https://docs.comma.ai
*   Learn more on the [community wiki](https://github.com/commaai/openpilot/wiki)

## Safety and Testing

openpilot prioritizes safety through:

*   Compliance with [ISO26262](https://en.wikipedia.org/wiki/ISO_26262) guidelines.
*   Automated software-in-the-loop [tests](.github/workflows/selfdrive_tests.yaml) on every commit.
*   Safety model implemented in C, with code rigor detailed at [code rigor](https://github.com/commaai/panda#code-rigor).
*   Hardware-in-the-loop tests for panda ([tests](https://github.com/commaai/panda/tree/master/tests/safety) and [Jenkinsfile](https://github.com/commaai/panda/blob/master/Jenkinsfile)).
*   Continuous testing with comma devices replaying routes.

## License

openpilot is released under the [MIT License](LICENSE).

**Disclaimer:** This is alpha-quality software for research purposes only. Use at your own risk and comply with all local laws and regulations. NO WARRANTY EXPRESSED OR IMPLIED.

## User Data and comma Account

By using openpilot, your driving data is uploaded to our servers to improve the system. You can access your data via [comma connect](https://connect.comma.ai/). You can also disable data collection if you wish.  By using this software, you agree to [our Privacy Policy](https://comma.ai/privacy).