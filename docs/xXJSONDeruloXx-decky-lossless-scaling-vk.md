# Lossless Scaling for Steam Deck: Enhance Your Gaming Experience

Unlock smoother, higher frame rates on your Steam Deck with the **Lossless Scaling** plugin, seamlessly integrating the power of frame generation. Check out the [original repo](https://github.com/xXJSONDeruloXx/decky-lossless-scaling-vk).

<p align="center">
   <img src="assets/decky-lossless-logo.png" alt="Lossless Scaling for Steam Deck Logo" width="200"/>
</p>

> **Important Note:** This is an unofficial community plugin and not officially supported by the creators of Lossless Scaling or lsfg-vk. For support, please use the [Decky Lossless Discord Channel](https://discord.gg/TwvHdVucC3).

## Key Features

*   **Simplified Installation:** Automatically installs and configures the `lsfg-vk` Vulkan layer for frame generation.
*   **Intuitive UI:** User-friendly interface for adjusting frame generation settings directly on your Steam Deck.
*   **Customizable Performance:** Fine-tune performance and quality with adjustable FPS multiplier, flow scale, and performance modes.
*   **HDR Support:** Enable HDR output for compatible games, enhancing visual fidelity.
*   **Real-Time Configuration:** Apply changes instantly without restarting your games.
*   **Experimental Features:** Access advanced options like present mode override and FPS limiting.
*   **Easy Uninstallation:** Removes all installed files cleanly when no longer needed.

## What is Lossless Scaling for Steam Deck?

This Decky plugin simplifies the process of using the `lsfg-vk` (Lossless Scaling Frame Generation Vulkan layer) on your Steam Deck. It provides a controller-friendly interface to configure and manage frame generation, enhancing the smoothness and visual experience of your games on Linux-based systems like SteamOS and Bazzite.

## Getting Started

1.  **Purchase Lossless Scaling:** Acquire [Lossless Scaling](https://store.steampowered.com/app/993090/Lossless_Scaling/) from the Steam store.
2.  **Download and Install the Plugin:**
    *   Download the "Lossless Scaling.zip" file from the [latest release](https://github.com/xXJSONDeruloXx/decky-lossless-scaling-vk/releases).
    *   In Game Mode, enable "Developer Mode" in Decky Loader settings.
    *   Go to the "Developer" tab and select "Install Plugin from Zip," then select the downloaded zip file.
3.  **Configure Settings:** Access the plugin from the Decky menu and configure your settings.
4.  **Apply Launch Option:** Add `~/lsfg %command%` to your game's launch options in Steam Properties, or use the provided "Launch Option Clipboard" button.
5.  **Launch Your Game:** Frame generation will activate automatically based on your plugin configuration.

## Configuration Options

The plugin offers a range of configuration options:

### Core Settings
*   **FPS Multiplier:** Choose between 2x, 3x, or 4x frame generation.
*   **Flow Scale:** Adjust motion estimation quality; lower values enhance performance, while higher values improve quality.
*   **Performance Mode:** Recommended for most games, uses a lighter processing model.
*   **HDR Mode:** Enable for games supporting HDR output.

### Experimental Features
*   **Present Mode Override:** Force specific Vulkan presentation modes for compatibility.
*   **Base FPS Limit:** Set a base framerate cap before the multiplier is applied (useful for DirectX games).

All settings (except base FPS limit) are saved automatically and applied immediately without requiring a game restart.

## Troubleshooting

**Frame generation not working?**

*   Ensure `~/lsfg %command%` is in your game's launch options.
*   Verify that the Lossless Scaling DLL was detected in the plugin.
*   Try enabling Performance Mode.
*   Ensure your game is running in fullscreen mode.

**Performance issues?**

*   Lower the Flow Scale.
*   Enable Performance Mode.
*   Reduce the FPS multiplier (e.g., from 4x to 2x or 3x).
*   Consider using the experimental FPS limit for DirectX games.

## Support and Community

For per-game feedback and community support, please join the [Decky Lossless Discord Channel](https://discord.gg/TwvHdVucC3)

## Credits

*   **[PancakeTAS](https://github.com/PancakeTAS/lsfg-vk)** for the lsfg-vk Vulkan layer.
*   **[Lossless Scaling](https://store.steampowered.com/app/993090/Lossless_Scaling/)** developers for the original frame generation technology.
*   **[Deck Wizard](https://www.youtube.com/@DeckWizard)** for the helpful video tutorial.
*   The **Decky Loader** team.
*   Community contributors and testers.