# InkyPi: Your Customizable E-Ink Display for the Raspberry Pi

<img src="./docs/images/inky_clock.jpg" alt="InkyPi Display" />

**Transform your Raspberry Pi into a beautiful, low-power display with InkyPi, showcasing the information you need, elegantly and without distractions. [See the original repo](https://github.com/fatihak/InkyPi)**

## Key Features

*   **Crisp, Paper-like Aesthetic:** Enjoy easy-on-the-eyes visuals with no glare or backlighting, mimicking the look of real paper.
*   **Web-Based Control:** Effortlessly configure and update your display from any device on your network using a user-friendly web interface.
*   **Minimize Distractions:** Focus on what matters with a display free of LEDs, noise, and unwanted notifications.
*   **Easy Setup for Beginners:**  Simple installation and configuration, perfect for beginners and experienced makers.
*   **Open-Source & Customizable:** Modify, extend, and create your own plugins to tailor InkyPi to your specific needs.
*   **Scheduled Content Playlists:**  Set up automated playlists to display different plugins at specific times throughout the day.

## Plugins Available

InkyPi offers a variety of plugins to display different kinds of information:

*   **Image Upload:** Display any image from your browser.
*   **Daily Newspaper/Comic:** View daily comics and front pages of major newspapers.
*   **Clock:** Display the time with customizable clock faces.
*   **AI Image/Text:** Generate images and dynamic text using OpenAI models.
*   **Weather:** Show current conditions and forecasts with customizable layouts.
*   **Calendar:** Visualize your calendar from Google, Outlook, or Apple Calendar.

New plugins are continuously being added!  Learn how to create your own by checking out the [Building InkyPi Plugins](./docs/building_plugins.md) documentation.

## Hardware Requirements

You'll need the following to get started:

*   **Raspberry Pi:** (4, 3, or Zero 2 W) - A 40-pin pre-soldered header is recommended.
*   **MicroSD Card:**  Minimum 8 GB (e.g., [this one](https://amzn.to/3G3Tq9W))
*   **E-Ink Display:**  Choose from the supported displays:
    *   **Pimoroni Inky Impression:**
        *   [13.3 Inch Display](https://collabs.shop/q2jmza)
        *   [7.3 Inch Display](https://collabs.shop/q2jmza)
        *   [5.7 Inch Display](https://collabs.shop/ns6m6m)
        *   [4 Inch Display](https://collabs.shop/cpwtbh)
    *   **Pimoroni Inky wHAT:**
        *   [4.2 Inch Display](https://collabs.shop/jrzqmf)
    *   **Waveshare e-Paper Displays:**
        *   Spectra 6 (E6) Full Color:  [4 inch](https://www.waveshare.com/4inch-e-paper-hat-plus-e.htm?&aff_id=111126) | [7.3 inch](https://www.waveshare.com/7.3inch-e-paper-hat-e.htm?&aff_id=111126) | [13.3 inch](https://www.waveshare.com/13.3inch-e-paper-hat-plus-e.htm?&aff_id=111126)
        *   Black and White: [7.5 inch](https://www.waveshare.com/7.5inch-e-paper-hat.htm?&aff_id=111126) | [13.3 inch](https://www.waveshare.com/13.3inch-e-paper-hat-k.htm?&aff_id=111126)
        *   See [Waveshare e-paper displays](https://www.waveshare.com/product/raspberry-pi/displays/e-paper.htm?&aff_id=111126) or visit their [Amazon store](https://amzn.to/3HPRTEZ) for additional models.  **Note:** IT8951-based displays are not supported. See the [Waveshare Display Support](#waveshare-display-support) section for compatibility details.
*   **Picture Frame or 3D Stand:** Explore community-provided options in [community.md](./docs/community.md).

**Affiliate Disclosure:** *Some links above are affiliate links.  Purchases made through these links may provide a commission that supports the project, at no extra cost to you.*

## Installation

Here's how to install InkyPi:

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/fatihak/InkyPi.git
    ```
2.  **Navigate to the Project Directory:**
    ```bash
    cd InkyPi
    ```
3.  **Run the Installation Script (using sudo):**
    ```bash
    sudo bash install/install.sh [-W <waveshare device model>]
    ```

    *   **For Inky Displays (Pimoroni):**
        ```bash
        sudo bash install/install.sh
        ```
    *   **For Waveshare Displays:**  Specify your display model after the `-W` option (e.g., `epd7in3f`).
        ```bash
        sudo bash install/install.sh -W epd7in3f
        ```

After installation, reboot your Raspberry Pi. The display will show the InkyPi splash screen.

**Important Notes:**

*   The installation script requires `sudo` privileges.
*   It's recommended to start with a fresh Raspberry Pi OS installation.
*   The script automatically enables the required SPI and I2C interfaces.
*   For more details, see [installation.md](./docs/installation.md). You can also check out [this YouTube tutorial](https://youtu.be/L5PvQj1vfC4).

## Updating InkyPi

Keep your InkyPi up-to-date with these steps:

1.  **Navigate to the Project Directory:**
    ```bash
    cd InkyPi
    ```
2.  **Fetch the Latest Changes:**
    ```bash
    git pull
    ```
3.  **Run the Update Script (using sudo):**
    ```bash
    sudo bash install/update.sh
    ```
    This ensures that all new code changes and any required dependencies are applied.

## Uninstalling InkyPi

To remove InkyPi, run:

```bash
sudo bash install/uninstall.sh
```

## Roadmap

The InkyPi project is constantly evolving! Here's a glimpse of what's planned:

*   More Plugins!
*   Modular layouts to combine plugins.
*   Button support with customizable actions.
*   Improved Web UI for mobile devices.

Check out the [public Trello board](https://trello.com/b/SWJYWqe4/inkypi) to see upcoming features and vote on your favorites!

## Waveshare Display Support

InkyPi supports Waveshare e-Paper displays, offering an alternative to Pimoroni's Inky screens.

**Compatibility:** This project has been tested with several Waveshare models.  **IT8951-based displays are not supported.**  **Displays smaller than 4 inches are not recommended** due to resolution limitations.

If your Waveshare display has a driver in the [Waveshare Python EPD library](https://github.com/waveshareteam/e-Paper/tree/master/RaspberryPi_JetsonNano/python/lib/waveshare_epd), it's likely compatible. Use the `-W` option during installation to specify your display model.

## License

Distributed under the GPL 3.0 License. See [LICENSE](./LICENSE) for details.

This project includes fonts and icons with separate licensing and attribution requirements.  See [Attribution](./docs/attribution.md) for more information.

## Troubleshooting

Check the [troubleshooting guide](./docs/troubleshooting.md) for solutions.  If you still need help, create an issue on the [GitHub Issues](https://github.com/fatihak/InkyPi/issues) page.

**Pi Zero W Users:**  See the [Known Issues during Pi Zero W Installation](./docs/troubleshooting.md#known-issues-during-pi-zero-w-installation) section in the troubleshooting guide.

## Sponsoring

Support InkyPi's continued development:

<p align="center">
<a href="https://github.com/sponsors/fatihak" target="_blank"><img src="https://user-images.githubusercontent.com/345274/133218454-014a4101-b36a-48c6-a1f6-342881974938.png" alt="Become a Patreon" height="35" width="auto"></a>
<a href="https://www.patreon.com/akzdev" target="_blank"><img src="https://c5.patreon.com/external/logo/become_a_patron_button.png" alt="Become a Patreon" height="35" width="auto"></a>
<a href="https://www.buymeacoffee.com/akzdev" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/default-orange.png" alt="Buy Me A Coffee" height="35" width="auto"></a>
</p>

## Acknowledgements

Explore these related projects:

*   [PaperPi](https://github.com/txoof/PaperPi) - Excellent project with Waveshare support
    *   Shoutout to @txoof for assistance with InkyPi's installation.
*   [InkyCal](https://github.com/aceinnolab/Inkycal) - Modular plugins for custom dashboards.
*   [PiInk](https://github.com/tlstommy/PiInk) - Inspiration for InkyPi's Flask web UI.
*   [rpi_weather_display](https://github.com/sjnims/rpi_weather_display) - Alternative e-ink weather dashboard.