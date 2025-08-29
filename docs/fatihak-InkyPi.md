# InkyPi: Your Customizable E-Ink Display for Raspberry Pi

[![InkyPi E-Ink Display](docs/images/inky_clock.jpg)](https://github.com/fatihak/InkyPi)

**Transform your Raspberry Pi into a sleek, low-power information hub with InkyPi, an open-source, customizable E-Ink display.**

[See the original repo](https://github.com/fatihak/InkyPi)

## Key Features

*   **Eye-Friendly Display:** Enjoy crisp, minimalist visuals with E-Ink's natural paper-like aesthetic, free from glare and backlights.
*   **Web-Based Control:** Effortlessly configure and update your display from any device on your network using the intuitive web interface.
*   **Minimal Distractions:** Focus on the content that matters with a display that offers zero LEDs, noise, or distracting notifications.
*   **Easy Setup:** Get up and running quickly with simple installation and configuration, perfect for beginners and experienced makers.
*   **Open-Source & Customizable:** Modify, extend, and create your own plugins thanks to InkyPi's open-source nature.
*   **Scheduled Playlists:** Display different content at various times of the day using customizable playlists.

## Plugins

InkyPi offers a range of plugins to display the information you need:

*   Image Upload: Display any image from your browser.
*   Daily Newspaper/Comic: View your favorite comics and front pages of major newspapers.
*   Clock: Customize your clock face.
*   AI Image/Text: Generate images and dynamic text using OpenAI's models.
*   Weather: Display current conditions and forecasts.
*   Calendar: Visualize your calendar from Google, Outlook, or Apple Calendar.

And more plugins are coming soon! Explore [Building InkyPi Plugins](./docs/building_plugins.md) for custom plugin development.

## Hardware

*   Raspberry Pi (4 | 3 | Zero 2 W) - [Pre Soldered Header recommended]
*   MicroSD Card (min 8 GB) - [Example Amazon Link](https://amzn.to/3G3Tq9W)
*   E-Ink Display:
    *   Inky Impression by Pimoroni
        *   [13.3 Inch Display](https://collabs.shop/q2jmza)
        *   [7.3 Inch Display](https://collabs.shop/q2jmza)
        *   [5.7 Inch Display](https://collabs.shop/ns6m6m)
        *   [4 Inch Display](https://collabs.shop/cpwtbh)
    *   Inky wHAT by Pimoroni
        *   [4.2 Inch Display](https://collabs.shop/jrzqmf)
    *   Waveshare e-Paper Displays
        *   Spectra 6 (E6) Full Color
            *   [4 inch](https://www.waveshare.com/4inch-e-paper-hat-plus-e.htm?&aff_id=111126)
            *   [7.3 inch](https://www.waveshare.com/7.3inch-e-paper-hat-e.htm?&aff_id=111126)
            *   [13.3 inch](https://www.waveshare.com/13.3inch-e-paper-hat-plus-e.htm?&aff_id=111126)
        *   Black and White
            *   [7.5 inch](https://www.waveshare.com/7.5inch-e-paper-hat.htm?&aff_id=111126)
            *   [13.3 inch](https://www.waveshare.com/13.3inch-e-paper-hat-k.htm?&aff_id=111126)
        *   Browse more models on [Waveshare e-paper displays](https://www.waveshare.com/product/raspberry-pi/displays/e-paper.htm?&aff_id=111126) or visit their [Amazon store](https://amzn.to/3HPRTEZ). **Note:** IT8951 based displays are **not** supported.  See [Waveshare e-Paper](#waveshare-display-support) for more information.
*   Picture Frame or 3D Stand - [Community resources available at community.md](./docs/community.md)

**Disclosure:** *Some links above are affiliate links. I may earn a commission from qualifying purchases, at no extra cost to you, which helps support this project.*

## Installation

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
    *   For Waveshare displays, specify your model:

        ```bash
        sudo bash install/install.sh -W <waveshare device model>
        ```

        (e.g., `sudo bash install/install.sh -W epd7in3f`)

After installation, reboot your Raspberry Pi.

*   Requires `sudo` privileges; start with a fresh Raspberry Pi OS installation recommended.
*   SPI and I2C interfaces will be enabled.
*   See [installation.md](./docs/installation.md) and [YouTube tutorial](https://youtu.be/L5PvQj1vfC4) for more information.

## Update

1.  Navigate to the project directory:

    ```bash
    cd InkyPi
    ```
2.  Fetch the latest changes:

    ```bash
    git pull
    ```
3.  Run the update script with `sudo`:

    ```bash
    sudo bash install/update.sh
    ```

## Uninstall

To uninstall InkyPi, run:

```bash
sudo bash install/uninstall.sh
```

## Roadmap

*   More plugins.
*   Modular layouts for mixing plugins.
*   Button support with customizable actions.
*   Improved web UI on mobile devices.

See the public [Trello board](https://trello.com/b/SWJYWqe4/inkypi) for upcoming features.

## Waveshare Display Support

InkyPi supports various Waveshare e-Paper displays.

*   **IT8951 controller displays are NOT supported.**
*   **Screens smaller than 4 inches are NOT recommended** due to limited resolution.
*   If your display has a driver in the [Waveshare Python EPD library](https://github.com/waveshareteam/e-Paper/tree/master/RaspberryPi_JetsonNano/python/lib/waveshare_epd), it's likely compatible.
*   Use the `-W` option during installation to specify your Waveshare display model (without the `.py` extension).

## License

Distributed under the GPL 3.0 License, see [LICENSE](./LICENSE).

This project utilizes fonts and icons with separate licensing and attribution requirements, see [Attribution](./docs/attribution.md).

## Troubleshooting

Check the [troubleshooting guide](./docs/troubleshooting.md).  If problems persist, create an issue on the [GitHub Issues](https://github.com/fatihak/InkyPi/issues) page.

*   [Known Issues during Pi Zero W Installation](./docs/troubleshooting.md#known-issues-during-pi-zero-w-installation)
*   [Troubleshooting Guide](./docs/troubleshooting.md)

## Sponsoring

Support the ongoing development of InkyPi:

<p align="center">
<a href="https://github.com/sponsors/fatihak" target="_blank"><img src="https://user-images.githubusercontent.com/345274/133218454-014a4101-b36a-48c6-a1f6-342881974938.png" alt="Become a Patreon" height="35" width="auto"></a>
<a href="https://www.patreon.com/akzdev" target="_blank"><img src="https://c5.patreon.com/external/logo/become_a_patron_button.png" alt="Become a Patreon" height="35" width="auto"></a>
<a href="https://www.buymeacoffee.com/akzdev" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/default-orange.png" alt="Buy Me A Coffee" height="35" width="auto"></a>
</p>

## Acknowledgements

*   [PaperPi](https://github.com/txoof/PaperPi) - shoutout to @txoof
*   [InkyCal](https://github.com/aceinnolab/Inkycal)
*   [PiInk](https://github.com/tlstommy/PiInk)
*   [rpi_weather_display](https://github.com/sjnims/rpi_weather_display)
```
Key improvements and SEO considerations:

*   **Hook:**  A compelling one-sentence description to grab attention.
*   **Clear Headings:**  Organized content for readability and SEO.
*   **Keyword Optimization:**  Included relevant keywords throughout (e.g., "E-Ink," "Raspberry Pi," "customizable," "plugins").
*   **Bulleted Lists:**  Easy-to-scan key features and plugin information.
*   **Strong Call to Action:**  Encouraged users to explore the project, and a link back to the repo.
*   **Concise Language:**  Removed unnecessary words and streamlined explanations.
*   **Internal Linking:**  Used anchor links to connect related sections (e.g., "Waveshare Display Support" link).
*   **Image with Alt Text:**  Added alt text to the image for accessibility and SEO.
*   **Structured Data:**  The Markdown is structured to make it easy for search engines to understand the content.
*   **Bolded Important Information:**  Highlighted key points (e.g., display sizes, warnings).
*   **Removed Duplication:**  Removed repetition from the original README, focusing on the most important information.
*   **Affiliate Disclosure Clarity:**  Made the affiliate disclosure more prominent.
*   **Improved Plugin Description:** More engaging descriptions of the plugins.