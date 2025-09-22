# openpilot: The Open Source Driver-Assistance System for Smarter, Safer Driving

**Upgrade your car's driving capabilities with openpilot, an open-source autonomous driving system that enhances driver assistance features in hundreds of supported vehicles.** [Learn more at the original repo](https://github.com/commaai/openpilot).

*   **Advanced Driver Assistance:** Openpilot provides features like adaptive cruise control, lane keeping assist, and automatic lane changes.
*   **Broad Vehicle Support:** Compatible with 300+ supported car models.
*   **Community Driven:** Benefit from an active community of developers and users.
*   **Easy Installation:** Install openpilot on a comma 3X device and supported vehicle.
*   **Continuous Improvement:** Data-driven development ensures ongoing feature enhancements and safety improvements.
*   **Open Source:**  Access, modify, and contribute to the code.

## Getting Started

To use openpilot, you'll need:

1.  **Comma 3X Device:** Available at [comma.ai/shop](https://comma.ai/shop/comma-3x).
2.  **Software:**  Enter the URL `openpilot.comma.ai` during the comma 3X setup.
3.  **Supported Car:** Ensure your car is on the list of [275+ supported vehicles](docs/CARS.md).
4.  **Car Harness:**  A [car harness](https://comma.ai/shop/car-harness) to connect your comma 3X to your car.

Detailed installation instructions are available at [comma.ai/setup](https://comma.ai/setup).

## Software Branches

Choose the branch that best suits your needs:

| Branch             | URL                       | Description                                                                          |
| ------------------ | ------------------------- | ------------------------------------------------------------------------------------ |
| `release3`         | openpilot.comma.ai         | The stable release branch.                                                          |
| `release3-staging` | openpilot-test.comma.ai    | Staging branch for early access to new releases.                                   |
| `nightly`          | openpilot-nightly.comma.ai | Development branch, may be unstable.                                                |
| `nightly-dev`      | installer.comma.ai/commaai/nightly-dev |  Development branch with experimental features.                                    |

## Contributing and Community

Join the openpilot community and contribute to its development:

*   **Community Discord:**  Connect with other users and developers on [Discord](https://discord.comma.ai).
*   **Contribute:**  Review the [contributing docs](docs/CONTRIBUTING.md) to get started.
*   **Documentation:** Comprehensive documentation is available at [docs.comma.ai](https://docs.comma.ai).
*   **Tools:** Explore the [openpilot tools](tools/) for development.
*   **Community Wiki:**  Find helpful information on the [community wiki](https://github.com/commaai/openpilot/wiki).

## Safety and Testing

openpilot is developed with safety in mind:

*   **ISO26262 Guidelines:** Adheres to ISO26262 standards; see [SAFETY.md](docs/SAFETY.md).
*   **Continuous Testing:** Software-in-the-loop tests run on every commit.
*   **Robust Code:** The safety model code is written in C within `panda`.
*   **Hardware-in-the-Loop Testing:**  Extensive hardware-in-the-loop and Jenkins test suites.
*   **Real-World Testing:**  Continuously tested in a dedicated testing environment.

## Licensing and Data Usage

*   **License:** openpilot is released under the [MIT license](LICENSE).
*   **Data Collection:** By default, driving data is uploaded to comma.ai's servers for model training and improvement. Users can disable data collection. Review [our Privacy Policy](https://comma.ai/privacy) for details.