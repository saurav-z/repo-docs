# InkyPi: Your Customizable E-Ink Display for a Paper-Like Experience

<img src="./docs/images/inky_clock.jpg" alt="InkyPi E-Ink Display Clock" width="500"/>

**Transform your Raspberry Pi into a beautiful, low-power display for the information you need with InkyPi.** [View the original repository on GitHub](https://github.com/fatihak/InkyPi).

## Key Features

*   **Paper-Like Aesthetic:** Enjoy crisp, minimalist visuals with no glare or backlighting, perfect for eye comfort.
*   **Web Interface:** Easily configure and update your display from any device on your network.
*   **Minimal Distractions:** Focus on the content you care about with no LEDs, noise, or notifications.
*   **Easy Setup:** Simple installation and configuration, ideal for beginners and experienced makers.
*   **Open Source & Customizable:** Modify, extend, and create your own plugins with our open-source project.
*   **Scheduled Playlists:** Display different plugins at specific times for automated content updates.

## Available Plugins

InkyPi offers a variety of plugins to display the information you need, with more being added regularly:

*   **Image Upload:** Display your own images directly.
*   **Daily Newspaper/Comic:** View daily comics and front pages of major newspapers.
*   **Clock:** Customize your clock faces.
*   **AI Image/Text:** Generate images and dynamic text using AI models (OpenAI).
*   **Weather:** Display current conditions and forecasts.
*   **Calendar:** Visualize your calendar from Google, Outlook, or Apple Calendar.

For documentation on creating custom plugins, see [Building InkyPi Plugins](./docs/building_plugins.md).

## Hardware Requirements

*   **Raspberry Pi:** (4 | 3 | Zero 2 W)
    *   Recommended to get 40 pin Pre Soldered Header
*   **MicroSD Card:** (min 8 GB) - Example: [Amazon Link](https://amzn.to/3G3Tq9W)
*   **E-Ink Display:**
    *   **Pimoroni Inky Impression Displays**
        *   [13.3 Inch Display](https://collabs.shop/q2jmza)
        *   [7.3 Inch Display](https://collabs.shop/q2jmza)
        *   [5.7 Inch Display](https://collabs.shop/ns6m6m)
        *   [4 Inch Display](https://collabs.shop/cpwtbh)
    *   **Pimoroni Inky wHAT Displays**
        *   [4.2 Inch Display](https://collabs.shop/jrzqmf)
    *   **Waveshare e-Paper Displays**
        *   Spectra 6 (E6) Full Color: [4 inch](https://www.waveshare.com/4inch-e-paper-hat-plus-e.htm?&aff_id=111126) [7.3 inch](https://www.waveshare.com/7.3inch-e-paper-hat-e.htm?&aff_id=111126) [13.3 inch](https://www.waveshare.com/13.3inch-e-paper-hat-plus-e.htm?&aff_id=111126)
        *   Black and White: [7.5 inch](https://www.waveshare.com/7.5inch-e-paper-hat.htm?&aff_id=111126) [13.3 inch](https://www.waveshare.com/13.3inch-e-paper-hat-k.htm?&aff_id=111126)
        *   Browse [Waveshare e-paper displays](https://www.waveshare.com/product/raspberry-pi/displays/e-paper.htm?&aff_id=111126) or their [Amazon store](https://amzn.to/3HPRTEZ) for additional models.
*   **Picture Frame or 3D Stand:** Explore community-submitted models and builds at [community.md](./docs/community.md).

**Affiliate Disclosure:** Please note that some links above are affiliate links.  As an Amazon Associate I earn from qualifying purchases.

## Installation

Follow these steps to get InkyPi up and running:

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/fatihak/InkyPi.git
    ```
2.  **Navigate to the Project Directory:**
    ```bash
    cd InkyPi
    ```
3.  **Run the Installation Script:**
    ```bash
    sudo bash install/install.sh [-W <waveshare device model>]
    ```

    *   **For Inky Displays:** (Pimoroni)
        ```bash
        sudo bash install/install.sh
        ```
    *   **For Waveshare Displays:**  Specify your display model using the `-W` option:
        ```bash
        sudo bash install/install.sh -W <waveshare_model>
        ```
        (e.g., `sudo bash install/install.sh -W epd7in3f`)

After installation, reboot your Raspberry Pi. The display will show the InkyPi splash screen.  Refer to [installation.md](./docs/installation.md) for detailed instructions, including Raspberry Pi OS imaging, and this [YouTube tutorial](https://youtu.be/L5PvQj1vfC4).

**Important Notes:**

*   The installation script requires `sudo` privileges.
*   It automatically enables necessary SPI and I2C interfaces.

## Updating InkyPi

Keep your InkyPi up-to-date with these steps:

1.  **Navigate to the Project Directory:**
    ```bash
    cd InkyPi
    ```
2.  **Fetch Latest Changes:**
    ```bash
    git pull
    ```
3.  **Run the Update Script:**
    ```bash
    sudo bash install/update.sh
    ```

## Uninstalling InkyPi

To remove InkyPi from your device, use this command:

```bash
sudo bash install/uninstall.sh
```

## Roadmap

We're constantly adding new features!  Planned improvements include:

*   More plugins
*   Modular layouts
*   Button support for custom actions
*   Improved Web UI for mobile devices

Stay informed and help shape InkyPi's future via the [Trello board](https://trello.com/b/SWJYWqe4/inkypi).

## Waveshare Display Support

InkyPi supports Waveshare e-Paper displays.  **IT8951 controller-based displays are NOT supported**, and screens smaller than 4 inches are not recommended.  If your display model has a driver in the Waveshare [Python EPD library](https://github.com/waveshareteam/e-Paper/tree/master/RaspberryPi_JetsonNano/python/lib/waveshare_epd), it should be compatible.  Use the `-W` option during installation to specify your display model (e.g., `-W epd7in3f`).

## License

Distributed under the GPL 3.0 License. See [LICENSE](./LICENSE) for details.

This project includes fonts and icons with separate licensing and attribution requirements. See [Attribution](./docs/attribution.md) for details.

## Troubleshooting & Support

Refer to the [troubleshooting guide](./docs/troubleshooting.md) for common issues. If you need further assistance, please create an issue on the [GitHub Issues](https://github.com/fatihak/InkyPi/issues) page.

**Pi Zero W users:**  Be aware of potential installation issues. See the troubleshooting guide for more information on known issues during Pi Zero W Installation.

## Sponsoring

Support the development of InkyPi!

<p align="center">
<a href="https://github.com/sponsors/fatihak" target="_blank"><img src="https://user-images.githubusercontent.com/345274/133218454-014a4101-b36a-48c6-a1f6-342881974938.png" alt="Become a Patreon" height="35" width="auto"></a>
<a href="https://www.patreon.com/akzdev" target="_blank"><img src="https://c5.patreon.com/external/logo/become_a_patron_button.png" alt="Become a Patreon" height="35" width="auto"></a>
<a href="https://www.buymeacoffee.com/akzdev" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/default-orange.png" alt="Buy Me A Coffee" height="35" width="auto"></a>
</p>

## Acknowledgements

Check out these related projects:

*   [PaperPi](https://github.com/txoof/PaperPi)
*   [InkyCal](https://github.com/aceinnolab/Inkycal)
*   [PiInk](https://github.com/tlstommy/PiInk)
*   [rpi\_weather\_display](https://github.com/sjnims/rpi_weather_display)