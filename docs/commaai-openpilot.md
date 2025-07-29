# Openpilot: Revolutionizing Driver Assistance Systems

**Openpilot is an open-source, advanced driver-assistance system (ADAS) that enhances the driving experience in over 300 supported vehicles.** ([Original Repo](https://github.com/commaai/openpilot))

## Key Features

*   **Open Source:** Benefit from community contributions and transparency.
*   **ADAS Enhancement:** Upgrade your car's existing driver assistance features.
*   **Wide Vehicle Support:** Compatible with 300+ vehicles.
*   **Easy Installation:** Simple setup with a comma 3/3X device and car harness.
*   **Continuous Improvement:** Regular updates and improvements from the community and comma.ai.
*   **Data-Driven Development:** Utilizing collected driving data to train better models and improve performance.

## Getting Started with Openpilot

To use openpilot, you will need the following:

1.  **Supported Device:** A comma 3/3X ([comma.ai/shop](https://comma.ai/shop/comma-3x)).
2.  **Software:** Install the release version using the URL `openpilot.comma.ai` during setup.
3.  **Supported Car:** Ensure your car is on the [supported cars list](docs/CARS.md).
4.  **Car Harness:** A compatible [car harness](https://comma.ai/shop/car-harness) to connect to your vehicle.

For detailed installation instructions, visit: [comma.ai/setup](https://comma.ai/setup)

## Branches

| Branch               | URL                                     | Description                                                                                                     |
| -------------------- | --------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| `release3`           | openpilot.comma.ai                       | Openpilot's release branch.                                                                                       |
| `release3-staging`   | openpilot-test.comma.ai                 | Staging branch for early access to new releases.                                                                 |
| `nightly`            | openpilot-nightly.comma.ai              | Bleeding-edge development branch; expect instability.                                                            |
| `nightly-dev`        | installer.comma.ai/commaai/nightly-dev  | Includes experimental development features for some cars.                                                         |
| `secretgoodopenpilot` | installer.comma.ai/commaai/secretgoodopenpilot | Preview branch from the autonomy team with new driving models merged earlier than master. |

## Contributing and Community

Openpilot thrives on community contributions.

*   Join the [Discord community](https://discord.comma.ai)
*   Contribute to the project by following the [contributing guidelines](docs/CONTRIBUTING.md).
*   Explore the [openpilot tools](tools/)
*   Access code documentation at https://docs.comma.ai
*   Find information on the [community wiki](https://github.com/commaai/openpilot/wiki)

## Careers and Bounties

*   Interested in working on openpilot? Check out [comma.ai/jobs#open-positions](https://comma.ai/jobs#open-positions).
*   Explore [bounties](https://comma.ai/bounties) for external contributors.

## Safety and Testing

*   Follows [ISO26262](https://en.wikipedia.org/wiki/ISO_26262) guidelines, see [SAFETY.md](docs/SAFETY.md).
*   Software-in-the-loop [tests](.github/workflows/selfdrive_tests.yaml) run on every commit.
*   The code enforcing the safety model lives in panda and is written in C, see [code rigor](https://github.com/commaai/panda#code-rigor) for more details.
*   panda has software-in-the-loop [safety tests](https://github.com/commaai/panda/tree/master/tests/safety).
*   Hardware-in-the-loop Jenkins test suite.
*   panda has additional hardware-in-the-loop [tests](https://github.com/commaai/panda/blob/master/Jenkinsfile).
*   Continuous testing with multiple devices replaying routes.

<details>
<summary>License</summary>

openpilot is released under the MIT license. See [LICENSE](LICENSE) for details.

*THIS IS ALPHA QUALITY SOFTWARE FOR RESEARCH PURPOSES ONLY. THIS IS NOT A PRODUCT.*
*YOU ARE RESPONSIBLE FOR COMPLYING WITH LOCAL LAWS AND REGULATIONS.*
*NO WARRANTY EXPRESSED OR IMPLIED.*
</details>

<details>
<summary>Data Collection and Privacy</summary>

By default, openpilot uploads driving data to our servers. You can access your data through [comma connect](https://connect.comma.ai/). We use your data to train better models and improve openpilot. You can disable data collection.

openpilot logs data, including cameras, CAN, GPS, IMU, and logs. Driver-facing camera and microphone are only logged if you opt-in.

By using openpilot, you agree to [our Privacy Policy](https://comma.ai/privacy), which explains our data usage.
</details>