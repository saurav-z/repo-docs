# openpilot: Open Source Driver Assistance System

**Upgrade your car's driver assistance system with openpilot, an open-source driving system that enhances safety and convenience for over 300 supported car models.**  ([Original Repo](https://github.com/commaai/openpilot))

<div align="center" style="text-align: center;">

### Key Features

*   **Enhanced Driver Assistance:** Provides advanced features like lane keeping, adaptive cruise control, and automatic emergency braking.
*   **Open Source:**  Benefit from community contributions and transparency.
*   **Wide Compatibility:** Supports 300+ car models, with more constantly being added.
*   **Easy Installation:** Utilize a comma 3/3X device and follow straightforward setup instructions.
*   **Continuous Improvement:** Data-driven development ensures continuous model improvements and updates.

### Getting Started

1.  **Requirements:**
    *   [comma 3/3X](https://comma.ai/shop/comma-3x) device.
    *   [Supported Car](docs/CARS.md).
    *   [Car Harness](https://comma.ai/shop/car-harness)
2.  **Software Installation:** Use the URL `openpilot.comma.ai` within your comma 3/3X setup to install the latest release.
3.  **Installation Instructions:** Detailed setup instructions are available at [comma.ai/setup](https://comma.ai/setup).

### Community & Resources

*   **Documentation:** [Docs](https://docs.comma.ai)
*   **Roadmap:** [Roadmap](https://docs.comma.ai/contributing/roadmap/)
*   **Contribute:** [Contribute](https://github.com/commaai/openpilot/blob/master/docs/CONTRIBUTING.md)
*   **Community Discord:** [Community](https://discord.comma.ai)
*   **Community Wiki:** [Wiki](https://github.com/commaai/openpilot/wiki)

### Branches

Choose the branch that best suits your needs:

| Branch           | URL                        | Description                                                                         |
|------------------|----------------------------|-------------------------------------------------------------------------------------|
| `release3`         | openpilot.comma.ai         | Release branch.                                                                    |
| `release3-staging` | openpilot-test.comma.ai    | Staging branch for early access to new releases.                                  |
| `nightly`          | openpilot-nightly.comma.ai | Bleeding edge development branch (unstable).                                         |
| `nightly-dev`      | installer.comma.ai/commaai/nightly-dev | Experimental development features for some cars.      |
| `secretgoodopenpilot` | installer.comma.ai/commaai/secretgoodopenpilot | Preview branch from the autonomy team where new driving models get merged earlier than master. |


### Contributing

openpilot thrives on community contributions.

*   Join the [community Discord](https://discord.comma.ai)
*   Check out [the contributing docs](docs/CONTRIBUTING.md)
*   Check out the [openpilot tools](tools/)
*   Code documentation lives at https://docs.comma.ai
*   Information about running openpilot lives on the [community wiki](https://github.com/commaai/openpilot/wiki)

### Safety and Testing

openpilot prioritizes safety and undergoes rigorous testing:

*   Adheres to [ISO26262](https://en.wikipedia.org/wiki/ISO_26262) guidelines (see [SAFETY.md](docs/SAFETY.md)).
*   Continuous software-in-the-loop tests on every commit ([workflows](.github/workflows/selfdrive_tests.yaml)).
*   Code enforcing the safety model is written in C in panda (see [code rigor](https://github.com/commaai/panda#code-rigor)).
*   Software-in-the-loop [safety tests](https://github.com/commaai/panda/tree/master/tests/safety) in panda.
*   Hardware-in-the-loop Jenkins test suite.
*   Hardware-in-the-loop [tests](https://github.com/commaai/panda/blob/master/Jenkinsfile).
*   Continuous testing with multiple devices replaying routes.

<details>
<summary>License</summary>

Released under the MIT License.
</details>

<details>
<summary>User Data and Privacy</summary>

By default, openpilot uploads driving data for model improvement.  Users can disable data collection. See [Privacy Policy](https://comma.ai/privacy).
</details>