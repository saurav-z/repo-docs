# InkyPi: Your Customizable E-Ink Display for Raspberry Pi

<img src="./docs/images/inky_clock.jpg" alt="InkyPi Display" width="400"/>

**Transform your Raspberry Pi into a stylish and energy-efficient display with InkyPi, a customizable E-Ink solution.** [View the original repository](https://github.com/fatihak/InkyPi).

## Key Features

*   **Crisp, Paper-Like Aesthetics:** Enjoy easy-on-the-eyes visuals with no glare or backlighting.
*   **Web-Based Interface:** Easily configure and update your display from any device on your network.
*   **Minimal Distractions:** Focus on the content you care about with no LEDs, noise, or notifications.
*   **Simple Setup:** Perfect for beginners and makers.
*   **Open Source & Customizable:** Modify, create your own plugins, and tailor InkyPi to your needs.
*   **Scheduled Playlists:** Display different content at designated times with customizable plugin schedules.

## Available Plugins

*   **Image Upload:** Display any image from your browser.
*   **Daily Newspaper/Comic:** Stay up-to-date with daily content.
*   **Clock:** Customize your clock faces.
*   **AI Image/Text:** Generate dynamic content with OpenAI.
*   **Weather:** Stay informed with current conditions and forecasts.
*   **Calendar:** Visualize your calendar from various services.

**And more plugins are coming soon!** Explore [Building InkyPi Plugins](./docs/building_plugins.md) to create your own.

## Hardware Requirements

*   **Raspberry Pi:** (4 | 3 | Zero 2 W) - *Recommended to get 40 pin Pre Soldered Header*
*   **MicroSD Card:** (min 8 GB) - [Example](https://amzn.to/3G3Tq9W)
*   **E-Ink Display:**
    *   **Pimoroni Inky Impression:**
        *   [13.3 Inch Display](https://collabs.shop/q2jmza)
        *   [7.3 Inch Display](https://collabs.shop/q2jmza)
        *   [5.7 Inch Display](https://collabs.shop/ns6m6m)
        *   [4 Inch Display](https://collabs.shop/cpwtbh)
    *   **Pimoroni Inky wHAT:**
        *   [4.2 Inch Display](https://collabs.shop/jrzqmf)
    *   **Waveshare e-Paper Displays:**
        *   Spectra 6 (E6) Full Color [4 inch](https://www.waveshare.com/4inch-e-paper-hat-plus-e.htm?&aff_id=111126) | [7.3 inch](https://www.waveshare.com/7.3inch-e-paper-hat-e.htm?&aff_id=111126) | [13.3 inch](https://www.waveshare.com/13.3inch-e-paper-hat-plus-e.htm?&aff_id=111126)
        *   Black and White [7.5 inch](https://www.waveshare.com/7.5inch-e-paper-hat.htm?&aff_id=111126) | [13.3 inch](https://www.waveshare.com/13.3inch-e-paper-hat-k.htm?&aff_id=111126)
        *   See [Waveshare e-paper displays](https://www.waveshare.com/product/raspberry-pi/displays/e-paper.htm?&aff_id=111126) or visit their [Amazon store](https://amzn.to/3HPRTEZ) for additional models. **Note: IT8951 based displays are not supported.**
*   **Picture Frame or 3D Stand:** See [community.md](./docs/community.md) for community-made designs.

**Affiliate Disclosure:** The links above are affiliate links. I may earn a commission from qualifying purchases.

## Installation

1.  Clone the repository:

    ```bash
    git clone https://github.com/fatihak/InkyPi.git
    ```

2.  Navigate to the project directory:

    ```bash
    cd InkyPi
    ```

3.  Run the installation script:

    *   For Inky displays:

        ```bash
        sudo bash install/install.sh
        ```

    *   For Waveshare displays:

        ```bash
        sudo bash install/install.sh -W <waveshare device model>
        ```

        (Replace `<waveshare device model>` with your display model, e.g., `epd7in3f`).
After the installation, reboot your Raspberry Pi.

For more details, see [installation.md](./docs/installation.md) and [this YouTube tutorial](https://youtu.be/L5PvQj1vfC4).

## Updating InkyPi

1.  Navigate to the project directory:

    ```bash
    cd InkyPi
    ```

2.  Fetch updates:

    ```bash
    git pull
    ```

3.  Run the update script:

    ```bash
    sudo bash install/update.sh
    ```

## Uninstalling InkyPi

```bash
sudo bash install/uninstall.sh
```

## Roadmap

*   Additional Plugins
*   Modular layouts
*   Button Support
*   Improved Web UI

Stay up-to-date with the progress on the [Trello board](https://trello.com/b/SWJYWqe4/inkypi).

## Waveshare Display Support

InkyPi supports Waveshare e-Paper displays. Use the `-W` option to specify your model during installation.

*   **Note:** IT8951 controller based displays are not supported.

## License

Distributed under the [GPL 3.0 License](./LICENSE).

See [Attribution](./docs/attribution.md) for font and icon licensing.

## Troubleshooting

Check the [troubleshooting guide](./docs/troubleshooting.md).

For issues, create an issue on [GitHub Issues](https://github.com/fatihak/InkyPi/issues).

## Sponsorship

Support InkyPi's development!

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
```
Key improvements and SEO considerations:

*   **Clear Headline:** Includes the main keyword "InkyPi" and a concise description.
*   **One-Sentence Hook:**  Provides an engaging introduction.
*   **Bulleted Lists:**  Emphasize key features and benefits.
*   **Keyword Optimization:** Repeatedly uses relevant keywords such as "E-Ink display," "Raspberry Pi," "customizable," and "plugins."
*   **Structured Headings:** Improves readability and SEO.
*   **Hardware Section:** Makes it easy for users to find necessary parts, with links.
*   **Installation and Update Sections:**  Well-organized instructions.
*   **Waveshare Display Support:** Clearly outlines compatibility and installation steps.
*   **Roadmap and Trello Link:**  Keeps users engaged.
*   **Call to Action (Sponsorship):** Encourages support.
*   **Acknowledgments:**  Provides context and gives credit.
*   **Link Back:**  Includes a link to the original repository.
*   **Alt Text:** Added alt text to the image
*   **Affiliate Disclosure:** Explicitly states the use of affiliate links.