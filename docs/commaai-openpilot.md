# openpilot: Open Source Driver Assistance System

**openpilot is an open-source, community-driven driver assistance system that enhances the capabilities of supported vehicles, transforming your driving experience.** ([Original Repo](https://github.com/commaai/openpilot))

*   **Enhance your Drive:** Upgrade the driver assistance system in over 300 supported car models.
*   **Open Source:** Benefit from community contributions, transparency, and continuous improvement.
*   **Advanced Features:** Access adaptive cruise control, lane keeping assist, and automatic lane centering.
*   **Easy Installation:** Quick setup with a comma 3/3X device and car harness.
*   **Active Community:** Engage with fellow users and developers on the Discord server.
*   **Regular Updates:** Stay up-to-date with the latest features and improvements.

## Getting Started with openpilot

To use openpilot in your car, you'll need:

1.  **Supported Device:** A comma 3/3X device available at [comma.ai/shop](https://comma.ai/shop/comma-3x).
2.  **Software:** Install the release version by entering the URL `openpilot.comma.ai` during setup.
3.  **Supported Car:** Check the [CARS.md](docs/CARS.md) file for a list of supported vehicles.
4.  **Car Harness:** Obtain a car harness from [comma.ai/shop/car-harness](https://comma.ai/shop/car-harness) to connect your device.

Detailed installation instructions are available at [comma.ai/setup](https://comma.ai/setup).

## Branches

Choose the right branch for your needs:

| Branch            | URL                                   | Description                                                                         |
| ----------------- | ------------------------------------- | ------------------------------------------------------------------------------------- |
| `release3`        | openpilot.comma.ai                     | Stable release branch.                                                                 |
| `release3-staging`| openpilot-test.comma.ai               | Staging branch for early access to new releases.                                      |
| `nightly`         | openpilot-nightly.comma.ai            | Development branch; may be unstable.                                                 |
| `nightly-dev`     | installer.comma.ai/commaai/nightly-dev | Nightly with experimental development features for some cars.                        |
| `secretgoodopenpilot` | installer.comma.ai/commaai/secretgoodopenpilot | Preview branch from the autonomy team with early driving models. |

## Contributing to openpilot

openpilot is developed by [comma](https://comma.ai/) and community members. We welcome contributions!

*   Join the [community Discord](https://discord.comma.ai).
*   Review the [contributing docs](docs/CONTRIBUTING.md).
*   Explore the [openpilot tools](tools/).
*   Find code documentation at https://docs.comma.ai.
*   Check out the [community wiki](https://github.com/commaai/openpilot/wiki).

[comma is hiring](https://comma.ai/jobs#open-positions) and offers bounties for contributors.

## Safety and Testing

*   openpilot adheres to [ISO26262](https://en.wikipedia.org/wiki/ISO_26262) guidelines.  See [SAFETY.md](docs/SAFETY.md) for more details.
*   Automated software-in-the-loop tests run on every commit.
*   The safety model code is written in C within `panda`.
*   `panda` has software-in-the-loop safety tests.
*   Hardware-in-the-loop tests are used internally.
*   Continuous testing on multiple devices.

<details>
<summary>MIT License</summary>

openpilot is released under the MIT license.  Refer to the license for details.
</details>

<details>
<summary>User Data and comma Account</summary>

openpilot collects driving data to improve the system. You can access your data through [comma connect](https://connect.comma.ai/). Users may disable data collection. By using openpilot, you agree to [our Privacy Policy](https://comma.ai/privacy).
</details>