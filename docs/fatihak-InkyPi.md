# InkyPi: Your Customizable E-Ink Display for Raspberry Pi

[![InkyPi Display](docs/images/inky_clock.jpg)](https://github.com/fatihak/InkyPi)

**Transform your Raspberry Pi into a low-power, distraction-free display with InkyPi, displaying the information you care about with a simple web interface.** [View the original repository](https://github.com/fatihak/InkyPi)

## Key Features

*   **Paper-like Aesthetic:** Enjoy crisp, easy-on-the-eyes visuals with no glare, thanks to the E-Ink display.
*   **Web-Based Control:** Easily configure and update your display from any device on your network via a user-friendly web interface.
*   **Minimize Distractions:** Eliminate LEDs, noise, and notifications, focusing on the content that matters.
*   **Simple Setup:** Perfect for beginners and makers, offering easy installation and configuration.
*   **Open Source & Customizable:** Modify, personalize, and create your own plugins to fit your needs.
*   **Scheduled Playlists:** Set up playlists to display different plugins at specified times.

## Plugins

InkyPi offers a wide range of plugins, with more constantly being developed:

*   **Image Upload:** Display any image you choose.
*   **Daily Newspaper/Comic:** Stay up-to-date with your favorite news and comics.
*   **Clock:** Customize your clock face with different styles and features.
*   **AI Image/Text:** Generate images and dynamic text using OpenAI models.
*   **Weather:** Get current weather conditions and forecasts with customizable layouts.
*   **Calendar:** Visualize your calendar from Google, Outlook, or Apple Calendar.

[Building InkyPi Plugins](./docs/building_plugins.md) for custom plugin development.

## Hardware Requirements

*   **Raspberry Pi:** (4 | 3 | Zero 2 W)
    *   Recommended: 40-pin Pre-Soldered Header
*   **MicroSD Card:** (min 8 GB), e.g., [this one](https://amzn.to/3G3Tq9W)
*   **E-Ink Display:**
    *   **Pimoroni Inky Impression:**
        *   [13.3 Inch Display](https://collabs.shop/q2jmza)
        *   [7.3 Inch Display](https://collabs.shop/q2jmza)
        *   [5.7 Inch Display](https://collabs.shop/ns6m6m)
        *   [4 Inch Display](https://collabs.shop/cpwtbh)
    *   **Pimoroni Inky wHAT:**
        *   [4.2 Inch Display](https://collabs.shop/jrzqmf)
    *   **Waveshare e-Paper Displays:**
        *   Spectra 6 (E6) Full Color: [4 inch](https://www.waveshare.com/4inch-e-paper-hat-plus-e.htm?&aff_id=111126), [7.3 inch](https://www.waveshare.com/7.3inch-e-paper-hat-e.htm?&aff_id=111126), [13.3 inch](https://www.waveshare.com/13.3inch-e-paper-hat-plus-e.htm?&aff_id=111126)
        *   Black and White: [7.5 inch](https://www.waveshare.com/7.5inch-e-paper-hat.htm?&aff_id=111126), [13.3 inch](https://www.waveshare.com/13.3inch-e-paper-hat-k.htm?&aff_id=111126)
        *   For more models, see [Waveshare e-paper displays](https://www.waveshare.com/product/raspberry-pi/displays/e-paper.htm?&aff_id=111126) or visit their [Amazon store](https://amzn.to/3HPRTEZ).  **Note:** IT8951-based displays are not supported. See [Waveshare e-Paper Support](#waveshare-display-support) for details.
*   **Picture Frame or 3D Stand:**
    *   See [community.md](./docs/community.md) for community submissions.

**Disclaimer:** Affiliate links are included, which may generate a commission for the project at no extra cost to you.

## Installation

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/fatihak/InkyPi.git
    ```
2.  **Navigate to the Project Directory:**
    ```bash
    cd InkyPi
    ```
3.  **Run the Installation Script with `sudo`:**

    *   For Inky displays:
        ```bash
        sudo bash install/install.sh
        ```
    *   For Waveshare displays, specify the model:
        ```bash
        sudo bash install/install.sh -W <waveshare device model>
        ```
        (e.g., `sudo bash install/install.sh -W epd7in3f`)

After installation, reboot your Raspberry Pi.  Refer to [installation.md](./docs/installation.md) and the [YouTube tutorial](https://youtu.be/L5PvQj1vfC4) for more details.

**Important Notes:**

*   Requires `sudo` privileges.  A fresh Raspberry Pi OS installation is recommended.
*   Automatically enables SPI and I2C interfaces.

## Updating InkyPi

1.  **Navigate to the Project Directory:**
    ```bash
    cd InkyPi
    ```
2.  **Fetch Latest Changes:**
    ```bash
    git pull
    ```
3.  **Run the Update Script with `sudo`:**
    ```bash
    sudo bash install/update.sh
    ```

## Uninstalling

To uninstall InkyPi:

```bash
sudo bash install/uninstall.sh
```

## Roadmap

Stay tuned for future enhancements!  Planned features include:

*   More plugins!
*   Modular layouts for plugin customization
*   Button support with custom actions
*   Improved mobile Web UI

Check out the [Trello board](https://trello.com/b/SWJYWqe4/inkypi) for upcoming features.

## Waveshare Display Support

InkyPi supports Waveshare e-Paper displays.  **IT8951-based displays are NOT supported.**  If your Waveshare model is supported by the [Waveshare Python EPD library](https://github.com/waveshareteam/e-Paper/tree/master/RaspberryPi_JetsonNano/python/lib/waveshare_epd), it should be compatible. Use the `-W` option in the installation script, specifying your display model.

## License

Distributed under the GPL 3.0 License. See [LICENSE](./LICENSE).

Fonts and icons have separate licensing; see [Attribution](./docs/attribution.md).

## Troubleshooting

Consult the [troubleshooting guide](./docs/troubleshooting.md) and the [GitHub Issues](https://github.com/fatihak/InkyPi/issues) page.

**Note for Pi Zero W Users:**  See [Known Issues during Pi Zero W Installation](./docs/troubleshooting.md#known-issues-during-pi-zero-w-installation).

## Sponsoring

Support the project's development:

<p align="center">
<a href="https://github.com/sponsors/fatihak" target="_blank"><img src="https://user-images.githubusercontent.com/345274/133218454-014a4101-b36a-48c6-a1f6-342881974938.png" alt="Become a Patreon" height="35" width="auto"></a>
<a href="https://www.patreon.com/akzdev" target="_blank"><img src="https://c5.patreon.com/external/logo/become_a_patron_button.png" alt="Become a Patreon" height="35" width="auto"></a>
<a href="https://www.buymeacoffee.com/akzdev" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/default-orange.png" alt="Buy Me A Coffee" height="35" width="auto"></a>
</p>

## Acknowledgements

Explore these related projects:

*   [PaperPi](https://github.com/txoof/PaperPi)
*   [InkyCal](https://github.com/aceinnolab/Inkycal)
*   [PiInk](https://github.com/tlstommy/PiInk)
*   [rpi_weather_display](https://github.com/sjnims/rpi_weather_display)