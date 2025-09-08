# InkyPi: Your Customizable E-Ink Display for Raspberry Pi

<img src="./docs/images/inky_clock.jpg" alt="InkyPi Clock Display" width="500"/>

**Transform your Raspberry Pi into a stylish and energy-efficient display with InkyPi, an open-source project for displaying the information you need on a crisp, paper-like E-Ink screen.** ([Back to Original Repo](https://github.com/fatihak/InkyPi))

## Key Features

*   **Eye-Friendly Display:** Enjoy a crisp, paper-like aesthetic with no glare or backlighting.
*   **Web-Based Configuration:** Easily set up and customize your display from any device on your network using the intuitive web interface.
*   **Minimalist Design:** Eliminate distractions with an E-Ink display that offers no LEDs, noise, or notifications, just the content you need.
*   **Easy Setup:** Get up and running quickly with simple installation and configuration, perfect for beginners and makers.
*   **Open Source & Customizable:** Modify, extend, and create your own plugins with the open-source nature of InkyPi.
*   **Scheduled Playlists:** Display various plugins with scheduled playlists, switching content at designated times.

## Available Plugins

InkyPi offers a variety of plugins to display the information that matters most to you:

*   **Image Upload:** Display any image from your browser.
*   **Daily Newspaper/Comic:** View daily comics and front pages from major newspapers.
*   **Clock:** Customize your time display with various clock faces.
*   **AI Image/Text:** Generate images and dynamic text using OpenAI's models.
*   **Weather:** Display current weather conditions and multi-day forecasts with a customizable layout.
*   **Calendar:** Visualize your calendar from Google, Outlook, or Apple Calendar with customizable layouts.

More plugins are on the way! Check out [Building InkyPi Plugins](./docs/building_plugins.md) to learn how to create your own.

## Hardware Requirements

To get started with InkyPi, you'll need the following hardware:

*   Raspberry Pi (4 | 3 | Zero 2 W) - A 40-pin pre-soldered header is recommended.
*   MicroSD Card (min 8 GB) - [Example on Amazon](https://amzn.to/3G3Tq9W)
*   E-Ink Display: Choose from the following supported displays:
    *   **Inky Impression by Pimoroni**
        *   [13.3 Inch Display](https://collabs.shop/q2jmza)
        *   [7.3 Inch Display](https://collabs.shop/q2jmza)
        *   [5.7 Inch Display](https://collabs.shop/ns6m6m)
        *   [4 Inch Display](https://collabs.shop/cpwtbh)
    *   **Inky wHAT by Pimoroni**
        *   [4.2 Inch Display](https://collabs.shop/jrzqmf)
    *   **Waveshare e-Paper Displays**
        *   Spectra 6 (E6) Full Color **[4 inch](https://www.waveshare.com/4inch-e-paper-hat-plus-e.htm?&aff_id=111126)** **[7.3 inch](https://www.waveshare.com/7.3inch-e-paper-hat-e.htm?&aff_id=111126)** **[13.3 inch](https://www.waveshare.com/13.3inch-e-paper-hat-plus-e.htm?&aff_id=111126)**
        *   Black and White **[7.5 inch](https://www.waveshare.com/7.5inch-e-paper-hat.htm?&aff_id=111126)** **[13.3 inch](https://www.waveshare.com/13.3inch-e-paper-hat-k.htm?&aff_id=111126)**
        *   See [Waveshare e-paper displays](https://www.waveshare.com/product/raspberry-pi/displays/e-paper.htm?&aff_id=111126) or visit their [Amazon store](https://amzn.to/3HPRTEZ) for additional models. Note that some models like the IT8951 based displays are not supported. See later section on [Waveshare e-Paper](#waveshare-display-support) compatibilty for more information.
*   Picture Frame or 3D Stand - Explore community-submitted models and builds in [community.md](./docs/community.md).

**Affiliate Disclosure:** Some of the links provided are affiliate links. If you make a purchase through these links, I may earn a commission at no extra cost to you, supporting the development of this project.

## Installation

Get started with InkyPi in a few simple steps:

1.  **Clone the Repository:**

    ```bash
    git clone https://github.com/fatihak/InkyPi.git
    ```

2.  **Navigate to the Project Directory:**

    ```bash
    cd InkyPi
    ```

3.  **Run the Installation Script:**

    *   **For Inky Displays:**

        ```bash
        sudo bash install/install.sh
        ```

    *   **For Waveshare Displays:**  Specify your display model after the `-W` option.

        ```bash
        sudo bash install/install.sh -W <waveshare device model>
        ```
        (e.g., `sudo bash install/install.sh -W epd7in3f`)

After the installation script completes, you'll be prompted to reboot your Raspberry Pi. Once rebooted, your InkyPi will display the splash screen and be ready for configuration.  Refer to [installation.md](./docs/installation.md) for detailed instructions, including imaging your microSD card with Raspberry Pi OS, and check out [this YouTube tutorial](https://youtu.be/L5PvQj1vfC4).

**Important Notes:**
* The installation script requires sudo privileges.  Start with a fresh Raspberry Pi OS installation to minimize potential conflicts.
*  The script automatically enables the required SPI and I2C interfaces.

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

3.  **Run the Update Script:**

    ```bash
    sudo bash install/update.sh
    ```

## Uninstalling InkyPi

Remove InkyPi from your Raspberry Pi using this command:

```bash
sudo bash install/uninstall.sh
```

## Roadmap

InkyPi is a continuously evolving project with exciting planned features:

*   Plugin Expansion
*   Modular Layouts
*   Button Support with Customizable Actions
*   Improved Web UI on Mobile

Contribute to the future of InkyPi! Check out the public [Trello board](https://trello.com/b/SWJYWqe4/inkypi) and vote on your favorite features.

## Waveshare Display Support

InkyPi also supports Waveshare e-Paper displays.  **Please note: Displays based on the IT8951 controller are not supported.  Screens smaller than 4 inches are not recommended.**

If your Waveshare display model has a corresponding driver within the [Waveshare Python EPD library](https://github.com/waveshareteam/e-Paper/tree/master/RaspberryPi_JetsonNano/python/lib/waveshare_epd), then it is likely compatible. Specify your Waveshare display model during installation with the `-W` option (without the `.py` extension). The script will automatically install the necessary driver.

## License

InkyPi is released under the GPL 3.0 License. For further details, please see the [LICENSE](./LICENSE) file.

The project also includes fonts and icons with their own licensing and attribution requirements. Please see [Attribution](./docs/attribution.md) for details.

## Troubleshooting

Find solutions to common issues in the [troubleshooting guide](./docs/troubleshooting.md). For more advanced problems, please create an issue on the [GitHub Issues](https://github.com/fatihak/InkyPi/issues) page.

**Note for Pi Zero W Users:** There are known installation issues. Refer to the [Known Issues during Pi Zero W Installation](./docs/troubleshooting.md#known-issues-during-pi-zero-w-installation) section.

## Sponsoring

Support the ongoing development of InkyPi! Your sponsorship helps ensure that the project continues to evolve and offer new features.

<p align="center">
<a href="https://github.com/sponsors/fatihak" target="_blank"><img src="https://user-images.githubusercontent.com/345274/133218454-014a4101-b36a-48c6-a1f6-342881974938.png" alt="Become a Patreon" height="35" width="auto"></a>
<a href="https://www.patreon.com/akzdev" target="_blank"><img src="https://c5.patreon.com/external/logo/become_a_patron_button.png" alt="Become a Patreon" height="35" width="auto"></a>
<a href="https://www.buymeacoffee.com/akzdev" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/default-orange.png" alt="Buy Me A Coffee" height="35" width="auto"></a>
</p>

## Acknowledgements

Check out these related projects:

*   [PaperPi](https://github.com/txoof/PaperPi) - Great project with Waveshare support.
    *   Shoutout to @txoof for InkyPi's installation help
*   [InkyCal](https://github.com/aceinnolab/Inkycal) - Modular plugins for custom dashboards.
*   [PiInk](https://github.com/tlstommy/PiInk) - Inspiration for the InkyPi web UI.
*   [rpi_weather_display](https://github.com/sjnims/rpi_weather_display) - An alternative e-ink weather dashboard.