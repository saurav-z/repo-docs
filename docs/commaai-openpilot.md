# openpilot: Open Source Driver Assistance for Your Car

**Transform your driving experience with openpilot, an open-source driving system that enhances the driver assistance capabilities of over 300 supported vehicles!**  Find the original project on [GitHub](https://github.com/commaai/openpilot).

## Key Features

*   **Enhanced Driver Assistance:** Upgrade your vehicle with advanced features like adaptive cruise control, lane keeping assist, and automatic lane changes.
*   **Open Source:** Benefit from community contributions, transparency, and the ability to customize the system to your needs.
*   **Wide Vehicle Support:**  Works with a growing list of over 300+ supported car models.
*   **Easy Installation:** Simple setup process using a comma 3/3X device.
*   **Continuous Improvement:** Driven by a community of developers and constantly updated with new features and improvements.

## Getting Started

To get started with openpilot:

1.  **Get a Compatible Device:** Purchase a [comma 3/3X](https://comma.ai/shop/comma-3x).
2.  **Install the Software:** Enter the URL `openpilot.comma.ai` in the device's software setup.
3.  **Check Vehicle Compatibility:** Ensure your car is on the [supported vehicle list](docs/CARS.md).
4.  **Connect the Hardware:** Use a [car harness](https://comma.ai/shop/car-harness) to connect your comma 3/3X to your car.
5.  **Follow Installation Instructions:**  Refer to the detailed [setup instructions](https://comma.ai/setup).

### Software Branches
| Branch           | URL                                    | Description                                                                         |
|------------------|----------------------------------------|-------------------------------------------------------------------------------------|
| `release3`         | openpilot.comma.ai                      | This is openpilot's release branch.                                                 |
| `release3-staging` | openpilot-test.comma.ai                | This is the staging branch for releases. Use it to get new releases slightly early. |
| `nightly`          | openpilot-nightly.comma.ai             | This is the bleeding edge development branch. Do not expect this to be stable.      |
| `nightly-dev`      | installer.comma.ai/commaai/nightly-dev | Same as nightly, but includes experimental development features for some cars.      |
| `secretgoodopenpilot` | installer.comma.ai/commaai/secretgoodopenpilot | This is a preview branch from the autonomy team where new driving models get merged earlier than master. |

## Contribute to openpilot

Join the openpilot community and help improve the project:

*   [Contribute](https://github.com/commaai/openpilot/blob/master/docs/CONTRIBUTING.md) to the code.
*   Join the [community Discord](https://discord.comma.ai).
*   Explore the [openpilot tools](tools/).
*   Read the [documentation](https://docs.comma.ai).
*   Find information on the [community wiki](https://github.com/commaai/openpilot/wiki).

**Interested in working on openpilot?** [comma is hiring](https://comma.ai/jobs#open-positions) and offers [bounties](https://comma.ai/bounties) for contributors.

## Safety and Testing

*   openpilot follows [ISO26262](https://en.wikipedia.org/wiki/ISO_26262) guidelines, see [SAFETY.md](docs/SAFETY.md) for details.
*   Automated [tests](.github/workflows/selfdrive_tests.yaml) run on every commit.
*   The safety model is written in C within panda, see [code rigor](https://github.com/commaai/panda#code-rigor) for details.
*   [Safety tests](https://github.com/commaai/panda/tree/master/tests/safety) are implemented in panda.
*   Hardware-in-the-loop Jenkins tests.
*   Continuous testing in a testing closet.

## License

openpilot is released under the [MIT License](LICENSE).

**Disclaimer:** This is alpha-quality software for research purposes only.  You are responsible for complying with local laws and regulations.  NO WARRANTY EXPRESSED OR IMPLIED.

## User Data and Privacy

By using openpilot, you agree that driving data is uploaded to comma.  You can access your data through [comma connect](https://connect.comma.ai/).  Data is used to improve the models. You can disable data collection.

Data logged includes road-facing cameras, CAN, GPS, IMU, magnetometer, thermal sensors, crashes, and operating system logs. The driver-facing camera and microphone are only logged if explicitly enabled.  By using openpilot, you agree to the [Privacy Policy](https://comma.ai/privacy).