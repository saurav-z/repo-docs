# InkyPi: Your Customizable E-Ink Display for Raspberry Pi

<img src="./docs/images/inky_clock.jpg" alt="InkyPi E-Ink Display" width="500"/>

**Transform your Raspberry Pi into a sleek, low-power display with InkyPi, the open-source solution for showcasing the information you need, effortlessly.** ([Back to the InkyPi Repository](https://github.com/fatihak/InkyPi))

## Key Features

*   **Eye-Friendly Display:** Enjoy a crisp, paper-like aesthetic with no glare or backlighting, perfect for any environment.
*   **Web-Based Control:** Easily configure and update your display from any device on your network using an intuitive web interface.
*   **Distraction-Free Viewing:** Focus on the content that matters most, without the interruptions of LEDs, noise, or notifications.
*   **Easy Setup & Customization:**  Simple installation and configuration, ideal for beginners and makers alike, with endless possibilities.
*   **Open Source & Extensible:** Modify, customize, and expand InkyPi with your own plugins to fit your unique needs.
*   **Scheduled Content Playlists:** Set up dynamic playlists to display different plugins at specific times, keeping your display fresh and informative.

## Plugins

InkyPi offers a variety of plugins to display various content.

*   **Image Upload:** Display custom images from your browser.
*   **Daily Newspaper/Comic:** Stay updated with daily comics and front pages from around the world.
*   **Clock:** Customize the time display with various clock faces.
*   **AI Image/Text:** Generate images and dynamic text using OpenAI's models.
*   **Weather:** Display current weather conditions and forecasts with a customizable layout.
*   **Calendar:** Visualize your calendar from Google, Outlook, or Apple Calendar.

More plugins are in development – see the [Building InkyPi Plugins](./docs/building_plugins.md) documentation for information on creating custom plugins.

## Hardware Requirements

*   **Raspberry Pi:** (4 | 3 | Zero 2 W) - Recommended to get 40 pin Pre Soldered Header
*   **MicroSD Card:** (min 8 GB) - [Recommended SD Card](https://amzn.to/3G3Tq9W)
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
        *   Browse more [Waveshare e-paper displays](https://www.waveshare.com/product/raspberry-pi/displays/e-paper.htm?&aff_id=111126) or visit their [Amazon store](https://amzn.to/3HPRTEZ). Note that some models like the IT8951 based displays are not supported. See the [Waveshare Display Support](#waveshare-display-support) section for compatibility.

*   **Picture Frame or 3D Stand:** Explore community-created designs in [community.md](./docs/community.md).

**Disclosure:**  *The above links are affiliate links. I may earn a commission from qualifying purchases made through them, at no extra cost to you, which helps support this project.*

## Installation

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/fatihak/InkyPi.git
    ```
2.  **Navigate to the Project Directory:**
    ```bash
    cd InkyPi
    ```
3.  **Run the Installation Script (with `sudo`):**
    ```bash
    sudo bash install/install.sh [-W <waveshare device model>]
    ```
    *   `-W <waveshare device model>`: **Use ONLY for Waveshare displays**. Specify the Waveshare model, e.g., `epd7in3f`.

    *   **Inky Displays (default):**
        ```bash
        sudo bash install/install.sh
        ```
    *   **Waveshare Displays:**
        ```bash
        sudo bash install/install.sh -W epd7in3f
        ```

After the installation, reboot your Raspberry Pi. The display will then show the InkyPi splash screen.

**Important Notes:**

*   The installation script requires `sudo` privileges.
*   A fresh Raspberry Pi OS installation is recommended to avoid conflicts.
*   SPI and I2C interfaces will be automatically enabled.
*   For detailed instructions, including microSD card imaging, see [installation.md](./docs/installation.md) or watch [this YouTube tutorial](https://youtu.be/L5PvQj1vfC4).

## Updating InkyPi

1.  **Navigate to the Project Directory:**
    ```bash
    cd InkyPi
    ```
2.  **Fetch the Latest Changes:**
    ```bash
    git pull
    ```
3.  **Run the Update Script (with `sudo`):**
    ```bash
    sudo bash install/update.sh
    ```

This process ensures that your InkyPi is up-to-date with the latest code and dependencies.

## Uninstalling InkyPi

```bash
sudo bash install/uninstall.sh
```

## Roadmap

InkyPi is continuously evolving. Planned features include:

*   More Plugins
*   Modular Layouts for Plugin Customization
*   Button Support with Customizable Actions
*   Improved Web UI on Mobile Devices

See the public [Trello board](https://trello.com/b/SWJYWqe4/inkypi) to explore upcoming features and vote on what you'd like to see next!

## Waveshare Display Support

InkyPi supports Waveshare e-Paper displays.  **IT8951 controller-based displays are NOT supported**, and screens under 4 inches are not recommended.

To install, use the `-W` option with the installation script, specifying your Waveshare display model. The script will then fetch and install the necessary drivers.

## License

Distributed under the GPL 3.0 License, see [LICENSE](./LICENSE) for details.

This project uses fonts and icons with separate licensing requirements. See [Attribution](./docs/attribution.md) for details.

## Troubleshooting

Consult the [troubleshooting guide](./docs/troubleshooting.md) for common issues and solutions.  If problems persist, create an issue on the [GitHub Issues](https://github.com/fatihak/InkyPi/issues) page.

**Note for Pi Zero W Users:**  Review the [Known Issues during Pi Zero W Installation](./docs/troubleshooting.md#known-issues-during-pi-zero-w-installation) section in the troubleshooting guide.

## Sponsoring

Support InkyPi's continued development.

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
*   [rpi_weather_display](https://github.com/sjnims/rpi_weather_display)
```
Key improvements and explanations:

*   **SEO Optimization:** Added keywords like "E-Ink display," "Raspberry Pi," "customizable," and plugin-specific terms (weather, calendar, etc.) throughout the text and headings.
*   **Concise Hook:**  A clear, attention-grabbing first sentence that summarizes the project's purpose.
*   **Clear Headings:** Improved organization with descriptive and well-formatted headings and subheadings.
*   **Bulleted Key Features:** Makes it easy to scan and understand the core benefits of InkyPi.
*   **Concise & Accurate Descriptions:** Rephrased sections for better clarity and readability.
*   **Call to Action:** Encouraged users to create issues and vote on upcoming features.
*   **Consistent Formatting:**  Used consistent markdown formatting (bold, italics, code blocks) throughout.
*   **Direct Links:** Included clickable links to the GitHub repository and other relevant resources.
*   **Targeted SEO:** Incorporated phrases that users might search for when looking for this type of project (e.g., "E-Ink display Raspberry Pi").
*   **Reorganized and Removed Redundancy:** Streamlined the content, removing some repetition and making the information more digestible.
*   **Clear Affiliate Disclosure:**  The affiliate disclosure is clearly labeled.
*   **Waveshare Section Enhanced:** Provided better clarity on Waveshare compatibility and installation.
*   **Sponsorship Callout:** Expanded the sponsorship callout to promote the different options.