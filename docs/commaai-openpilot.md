# openpilot: Open Source Driver Assistance for Your Car

**Upgrade your driving experience with openpilot, the open-source driving agent that adds advanced driver-assistance features to a wide range of vehicles.**  Learn more and contribute at the [original openpilot repository](https://github.com/commaai/openpilot).

## Key Features

*   **Enhanced Driver Assistance:** Adds features like adaptive cruise control, lane keeping assist, and automatic emergency braking to supported vehicles.
*   **Wide Vehicle Compatibility:** Works with over 300+ supported car models.
*   **Open Source & Community Driven:**  Benefit from a collaborative development model with contributions from comma.ai and a vibrant community.
*   **Regular Updates:** Continuously improved with new features, bug fixes, and expanded vehicle support.
*   **Easy Installation:**  Simple setup with a comma 3X device and harness, utilizing a straightforward software installation process.

## How to Get Started

To use openpilot, you'll need the following:

1.  **Supported Device:** A comma 3X, available at [comma.ai/shop](https://comma.ai/shop/comma-3x).
2.  **Software:** Install the openpilot release branch using the URL `openpilot.comma.ai` when prompted by the comma 3X.
3.  **Supported Car:** One of the [275+ supported cars](docs/CARS.md).
4.  **Car Harness:** A [car harness](https://comma.ai/shop/car-harness) to connect your comma 3X.

Detailed instructions for setup are available at [comma.ai/setup](https://comma.ai/setup).

## Branches

| Branch             | URL                      | Description                                                                              |
|--------------------|--------------------------|------------------------------------------------------------------------------------------|
| `release3`           | openpilot.comma.ai        | Stable release branch.                                                                  |
| `release3-staging`   | openpilot-test.comma.ai   | Staging branch for previewing upcoming releases.                                        |
| `nightly`            | openpilot-nightly.comma.ai | Bleeding-edge development branch, may be unstable.                                     |
| `nightly-dev`        | installer.comma.ai/commaai/nightly-dev | Nightly branch with experimental development features for some cars.           |

## Contributing

openpilot thrives on community contributions. Get involved by:

*   Joining the [community Discord](https://discord.comma.ai).
*   Reviewing the [contributing docs](docs/CONTRIBUTING.md).
*   Exploring the [openpilot tools](tools/).
*   Consulting the code documentation at https://docs.comma.ai.
*   Visiting the [community wiki](https://github.com/commaai/openpilot/wiki) for additional information.

## Safety and Testing

*   openpilot adheres to [ISO26262](https://en.wikipedia.org/wiki/ISO_26262) guidelines, details in [SAFETY.md](docs/SAFETY.md).
*   Continuous software-in-the-loop [tests](.github/workflows/selfdrive_tests.yaml) on every commit.
*   Safety model implemented in panda and written in C, see [code rigor](https://github.com/commaai/panda#code-rigor).
*   Panda includes software-in-the-loop [safety tests](https://github.com/commaai/panda/tree/master/tests/safety).
*   Extensive hardware-in-the-loop testing with a Jenkins test suite.
*   Dedicated testing environment with multiple devices continuously replaying routes.

## Legal & Data
*   Released under the [MIT License](LICENSE).
*   By using openpilot, you agree to [our Privacy Policy](https://comma.ai/privacy).
*   Openpilot collects driving data to improve the models. Data collection can be disabled.