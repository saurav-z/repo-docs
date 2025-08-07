# openpilot: Your Open-Source Autonomous Driving Co-Pilot

openpilot is a cutting-edge open-source software project that upgrades your car's driver-assistance system, offering advanced features like lane keeping and adaptive cruise control. Check out the [original repo](https://github.com/commaai/openpilot) for more information.

## Key Features

*   **Advanced Driver-Assistance:** Experience features like lane keeping, adaptive cruise control, and more.
*   **Wide Car Compatibility:** Supports over 300 car models.
*   **Open Source & Community Driven:** Contribute to the development and improvement of openpilot.
*   **Continuous Improvement:** Benefit from ongoing updates and model training.
*   **Data-Driven Development:** Contribute data to improve the model and the openpilot software.

## Getting Started

To use openpilot, you'll need:

1.  **Supported Device:** A comma 3/3X, available at [comma.ai/shop](https://comma.ai/shop/comma-3x).
2.  **Software:** Install the release version using the URL `openpilot.comma.ai`.
3.  **Supported Car:** Ensure your car is on the [supported car list](docs/CARS.md).
4.  **Car Harness:** A [car harness](https://comma.ai/shop/car-harness) to connect your device to your car.

Detailed setup instructions are available at [comma.ai/setup](https://comma.ai/setup).

## Branches

Select the right software build for you.

| Branch           | URL                         | Description                                                                         |
|------------------|-----------------------------|-------------------------------------------------------------------------------------|
| `release3`       | openpilot.comma.ai           | Release branch for stable use.                                                 |
| `release3-staging` | openpilot-test.comma.ai      | Staging branch for early access to new releases. |
| `nightly`          | openpilot-nightly.comma.ai    | Bleeding edge development branch.  |
| `nightly-dev`      | installer.comma.ai/commaai/nightly-dev | Includes experimental development features for some cars.     |
| `secretgoodopenpilot` | installer.comma.ai/commaai/secretgoodopenpilot | Preview branch from the autonomy team where new driving models get merged earlier than master. |

## Contribute & Connect

*   Join the [community Discord](https://discord.comma.ai)
*   Contribute by checking out the [contributing docs](docs/CONTRIBUTING.md)
*   Find the [openpilot tools](tools/)
*   Browse the [code documentation](https://docs.comma.ai)
*   Access information on the [community wiki](https://github.com/commaai/openpilot/wiki)

## Safety and Testing

*   Adheres to [ISO26262](https://en.wikipedia.org/wiki/ISO_26262) guidelines. See [SAFETY.md](docs/SAFETY.md) for details.
*   Continuous software-in-the-loop [tests](.github/workflows/selfdrive_tests.yaml).
*   Safety model code in panda, written in C, see [code rigor](https://github.com/commaai/panda#code-rigor) for details.
*   Software-in-the-loop [safety tests](https://github.com/commaai/panda/tree/master/tests/safety) in panda.
*   Hardware-in-the-loop Jenkins test suite.
*   Additional hardware-in-the-loop [tests](https://github.com/commaai/panda/blob/master/Jenkinsfile).
*   Continuous testing on multiple devices.

---

<details>
<summary>License</summary>

Openpilot is released under the MIT license. See [LICENSE](LICENSE) for the full text.

</details>

<details>
<summary>Data Usage</summary>

By default, openpilot uploads driving data to comma.ai servers. You can access your data through [comma connect](https://connect.comma.ai/). You can disable data collection if you wish.
</details>