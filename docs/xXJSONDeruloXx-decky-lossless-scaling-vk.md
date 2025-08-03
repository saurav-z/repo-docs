# Lossless Scaling for Steam Deck: Enhance Your Gaming Experience

**Unlock smoother frame rates and stunning visuals on your Steam Deck with the Lossless Scaling plugin, bringing the power of frame generation to your favorite games.**

[Check out the original repository](https://github.com/xXJSONDeruloXx/decky-lossless-scaling-vk)

---

## Key Features

This plugin simplifies the installation and configuration of `lsfg-vk` (Lossless Scaling Frame Generation Vulkan layer) on your Steam Deck, offering a user-friendly way to leverage frame generation technology.

*   **Simplified Installation:**  Automates the download and setup of `lsfg-vk`.
*   **Controller-Friendly UI:** Provides an intuitive interface within SteamOS and Bazzite for easy configuration.
*   **Seamless Integration:** Integrates with the existing **Lossless Scaling** application.
*   **Real-time Configuration:**  Changes to your settings are applied instantly without needing to restart the game.
*   **Customizable Settings:**
    *   **FPS Multiplier:** Choose between 2x, 3x, or 4x frame generation.
    *   **Flow Scale:** Adjust motion estimation for performance vs. quality.
    *   **Performance Mode:** Optimizes performance with a lighter processing model.
    *   **HDR Mode:** Enables HDR output for supported games.
    *   **Experimental Features:**  Override presentation modes and set base FPS limits.

---

## Installation

1.  **Download the plugin:** Get the "Lossless Scaling.zip" file from the [latest release](https://github.com/xXJSONDeruloXx/decky-lossless-scaling-vk/releases)
2.  **Install via Decky Loader:**
    *   Enable "Developer Mode" in Decky Loader settings.
    *   Go to the "Developer" tab and select "Install Plugin from Zip."
    *   Choose the downloaded "Lossless Scaling.zip" file.

---

## How to Use

1.  **Purchase and install Lossless Scaling:**  Get the Lossless Scaling application from the [Steam store](https://store.steampowered.com/app/993090/Lossless_Scaling/).
2.  **Open the plugin:** Access it from the Decky Loader menu.
3.  **Install lsfg-vk:** Click "Install lsfg-vk" within the plugin.
4.  **Configure Settings:** Adjust FPS multiplier, flow scale, and other settings in the UI.
5.  **Apply Launch Option:** Add `~/lsfg %command%` to your game's launch options.
6.  **Launch Your Game:** Frame generation will activate automatically using your chosen configuration.

---

## Troubleshooting

*   **Frame Generation Not Working?**
    *   Ensure you've added the correct launch option: `~/lsfg %command%`
    *   Verify that the Lossless Scaling DLL was detected correctly.
    *   Try enabling Performance Mode.
    *   Ensure the game is running in fullscreen mode.
*   **Performance Issues?**
    *   Lower the "Flow Scale" setting.
    *   Enable "Performance Mode."
    *   Reduce the FPS multiplier (e.g., from 4x to 2x or 3x).
    *   Consider using the experimental FPS limit feature.

---

## Feedback and Support

Join the [Decky Lossless Discord Channel](https://discord.gg/TwvHdVucC3) for community support and game-specific feedback.

---