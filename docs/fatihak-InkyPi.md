# InkyPi: Your Customizable E-Ink Display

**Transform your Raspberry Pi into a stylish and energy-efficient information hub with InkyPi.** ([Original Repo](https://github.com/fatihak/InkyPi))

<img src="./docs/images/inky_clock.jpg" alt="InkyPi Clock Display" />

## About InkyPi

InkyPi is an open-source project that lets you create a personalized e-ink display using a Raspberry Pi. Enjoy a distraction-free experience with its crisp, paper-like visuals, perfect for displaying the information you care about most.  With a simple web interface, setup and customization are a breeze!

**Key Features:**

*   **Eye-Friendly Display:** Experience crisp, glare-free visuals with a natural paper-like aesthetic.
*   **Web-Based Configuration:** Effortlessly update and configure your display from any device on your network.
*   **Minimize Distractions:** Eliminate LEDs, noise, and notifications; focus solely on the content you want.
*   **Beginner-Friendly:** Easy installation and configuration for makers of all skill levels.
*   **Open Source & Customizable:** Modify, extend, and build your own plugins.
*   **Scheduled Playlists:** Display different content with scheduled playlists.

## Plugins

InkyPi offers a variety of plugins to display the information you need. New plugins are constantly being added!

*   **Image Upload:** Display any image from your browser.
*   **Daily Newspaper/Comic:** View daily comics and front pages of major newspapers.
*   **Clock:** Customize your clock faces.
*   **AI Image/Text:** Generate images and dynamic text using OpenAI's models.
*   **Weather:** Display current weather conditions and forecasts.
*   **Calendar:** Visualize your calendar from Google, Outlook, or Apple Calendar.

For documentation on creating custom plugins, see [Building InkyPi Plugins](./docs/building_plugins.md).

## Hardware

The following hardware is required to run InkyPi:

*   Raspberry Pi (4 | 3 | Zero 2 W)
    *   Recommended to get 40 pin Pre Soldered Header
*   MicroSD Card (min 8 GB) ([Example](https://amzn.to/3G3Tq9W))
*   E-Ink Display:
    *   Inky Impression by Pimoroni
        *   [13.3 Inch Display](https://collabs.shop/q2jmza)
        *   [7.3 Inch Display](https://collabs.shop/q2jmza)
        *   [5.7 Inch Display](https://collabs.shop/ns6m6m)
        *   [4 Inch Display](https://collabs.shop/cpwtbh)
    *   Inky wHAT by Pimoroni
        *   [4.2 Inch Display](https://collabs.shop/jrzqmf)
    *   Waveshare e-Paper Displays
        *   Spectra 6 (E6) Full Color [4 inch](https://www.waveshare.com/4inch-e-paper-hat-plus-e.htm?&aff_id=111126) [7.3 inch](https://www.waveshare.com/7.3inch-e-paper-hat-e.htm?&aff_id=111126) [13.3 inch](https://www.waveshare.com/13.3inch-e-paper-hat-plus-e.htm?&aff_id=111126)
        *   Black and White [7.5 inch](https://www.waveshare.com/7.5inch-e-paper-hat.htm?&aff_id=111126) [13.3 inch](https://www.waveshare.com/13.3inch-e-paper-hat-k.htm?&aff_id=111126)
        *   See [Waveshare e-paper displays](https://www.waveshare.com/product/raspberry-pi/displays/e-paper.htm?&aff_id=111126) or visit their [Amazon store](https://amzn.to/3HPRTEZ) for additional models.

*   Picture Frame or 3D Stand
    *   See [community.md](./docs/community.md) for community submissions

**Disclaimer:** Some links above are affiliate links, and may earn a commission.

## Installation

Get started with InkyPi in a few simple steps:

1.  Clone the repository:

    ```bash
    git clone https://github.com/fatihak/InkyPi.git
    ```

2.  Navigate to the project directory:

    ```bash
    cd InkyPi
    ```

3.  Run the installation script with `sudo`:

    *   For Inky displays:

        ```bash
        sudo bash install/install.sh
        ```

    *   For [Waveshare displays](#waveshare-display-support):  Specify your display model:

        ```bash
        sudo bash install/install.sh -W <waveshare device model>
        ```

    (Replace `<waveshare device model>` with your display's model, e.g., `epd7in3f`).

After installation, reboot your Raspberry Pi. The display will then show the InkyPi splash screen.

**Important Notes:**

*   The installation script requires `sudo` privileges.
*   We recommend a fresh Raspberry Pi OS installation to avoid conflicts.
*   The script automatically enables necessary SPI and I2C interfaces.
*   Refer to [installation.md](./docs/installation.md) for detailed instructions and [this YouTube tutorial](https://youtu.be/L5PvQj1vfC4).

## Update

To keep your InkyPi up-to-date:

1.  Navigate to the project directory:

    ```bash
    cd InkyPi
    ```

2.  Fetch the latest changes:

    ```bash
    git pull
    ```

3.  Run the update script:

    ```bash
    sudo bash install/update.sh
    ```

## Uninstall

To remove InkyPi:

```bash
sudo bash install/uninstall.sh
```

## Roadmap

Explore the future of InkyPi:

*   Additional Plugins
*   Modular layouts for mixing and matching plugins.
*   Support for buttons with customizable action bindings.
*   Improved Web UI on mobile devices

Check out the public [Trello board](https://trello.com/b/SWJYWqe4/inkypi) to contribute and vote on new features!

## Waveshare Display Support

InkyPi offers support for Waveshare e-Paper displays.  **Note:** Displays based on the IT8951 controller are not supported, and screens smaller than 4 inches are not recommended.

See [Waveshare e-paper displays](https://www.waveshare.com/product/raspberry-pi/displays/e-paper.htm?&aff_id=111126) for additional models.

When installing, specify your Waveshare model with the `-W` option. The script will automatically install the necessary driver.

## License

Distributed under the GPL 3.0 License. See [LICENSE](./LICENSE) for details.

## Attribution

This project includes fonts and icons with separate licensing. See [Attribution](./docs/attribution.md) for details.

## Troubleshooting & Issues

Check the [troubleshooting guide](./docs/troubleshooting.md). If you're still having problems, report an issue on the [GitHub Issues](https://github.com/fatihak/InkyPi/issues) page.

## Sponsoring

Support the continued development of InkyPi:

<p align="center">
<a href="https://github.com/sponsors/fatihak" target="_blank"><img src="https://user-images.githubusercontent.com/345274/133218454-014a4101-b36a-48c6-a1f6-342881974938.png" alt="Become a Patreon" height="35" width="auto"></a>
<a href="https://www.patreon.com/akzdev" target="_blank"><img src="https://c5.patreon.com/external/logo/become_a_patron_button.png" alt="Become a Patreon" height="35" width="auto"></a>
<a href="https://www.buymeacoffee.com/akzdev" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/default-orange.png" alt="Buy Me A Coffee" height="35" width="auto"></a>
</p>

## Acknowledgements

*   [PaperPi](https://github.com/txoof/PaperPi)
*   [InkyCal](https://github.com/aceinnolab/Inkycal)
*   [PiInk](https://github.com/tlstommy/PiInk)
*   [rpi_weather_display](https://github.com/sjnims/rpi_weather_display)