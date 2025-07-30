# Openpilot: Enhance Your Ride with Advanced Driver Assistance

**Openpilot is an open-source driving agent that upgrades driver assistance systems in over 300 supported car models.** ([Original Repository](https://github.com/commaai/openpilot))

## Key Features:

*   **Upgrades Driver Assistance:** Improves existing features like adaptive cruise control and lane keeping assist.
*   **Open Source:**  Leverage and contribute to a community-driven project.
*   **Extensive Car Support:** Compatible with 300+ car models.
*   **Community Driven:** Join the active [Discord community](https://discord.comma.ai) for support and collaboration.
*   **Continuously Improving:** Benefit from ongoing development and updates.

## Getting Started

To use openpilot, you'll need:

1.  **Compatible Hardware:** A comma 3/3X device, available at [comma.ai/shop](https://comma.ai/shop/comma-3x).
2.  **Software Installation:** Install openpilot by entering the URL `openpilot.comma.ai` during the setup of your comma 3/3X.
3.  **Supported Vehicle:** A car from the list of [275+ supported cars](docs/CARS.md).
4.  **Car Harness:** A [car harness](https://comma.ai/shop/car-harness) to connect your device to your car.

## Openpilot Branches

Choose the branch that best suits your needs:

| Branch              | URL                        | Description                                                                         |
|---------------------|----------------------------|-------------------------------------------------------------------------------------|
| `release3`          | openpilot.comma.ai         | Stable release branch.                                                             |
| `release3-staging`  | openpilot-test.comma.ai    | Staging branch for early access to new releases.                                     |
| `nightly`           | openpilot-nightly.comma.ai | Bleeding-edge development branch (may be unstable).                                 |
| `nightly-dev`       | installer.comma.ai/commaai/nightly-dev | Includes experimental development features.      |
| `secretgoodopenpilot`       | installer.comma.ai/commaai/secretgoodopenpilot | Preview branch for new driving models. |

## Contributing and Community

Openpilot thrives on community contributions. Here's how you can get involved:

*   Join the [community Discord](https://discord.comma.ai)
*   Review the [contributing docs](docs/CONTRIBUTING.md)
*   Explore the [openpilot tools](tools/)
*   Access code documentation at https://docs.comma.ai
*   Learn more about running openpilot on the [community wiki](https://github.com/commaai/openpilot/wiki)

## Safety and Testing

Openpilot is built with safety in mind:

*   Adheres to [ISO26262](https://en.wikipedia.org/wiki/ISO_26262) guidelines (see [SAFETY.md](docs/SAFETY.md)).
*   Uses software-in-the-loop [tests](.github/workflows/selfdrive_tests.yaml) on every commit.
*   Code enforcing the safety model is written in C.
*   Includes [safety tests](https://github.com/commaai/panda/tree/master/tests/safety) for the panda component.
*   Employs a hardware-in-the-loop Jenkins test suite.

<details>
<summary>MIT License</summary>

Openpilot is released under the MIT license. (See original README for full details).
</details>

<details>
<summary>User Data and Privacy</summary>

Openpilot collects driving data to improve its models.  Users have the option to disable data collection.  (See original README for full details).
</details>