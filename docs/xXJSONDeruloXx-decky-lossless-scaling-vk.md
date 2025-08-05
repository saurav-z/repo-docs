# Enhance Your Steam Deck Gaming with Lossless Scaling & Frame Generation

Unlock the full potential of your Steam Deck with the **Decky Lossless Scaling Plugin**, bringing the power of frame generation and performance optimization to your favorite games. [Visit the original repository on GitHub](https://github.com/xXJSONDeruloXx/decky-lossless-scaling-vk) for more information.

## Key Features

*   **Effortless Installation:** Streamlines the setup of the `lsfg-vk` Vulkan layer for Lossless Scaling, allowing you to easily use frame generation.
*   **User-Friendly Interface:** Configure frame generation settings directly within the Decky Loader interface.
*   **Customizable Frame Generation:** Adjust the FPS multiplier (2x, 3x, or 4x) and flow scale to optimize performance and visual quality.
*   **Performance Modes:** Choose from performance and HDR modes to optimize your gameplay experience.
*   **Hot-Reloading:** Changes to your settings take effect instantly without requiring a game restart.
*   **Experimental Features:** Override presentation modes and set base FPS limits for enhanced compatibility.
*   **Automatic Detection:** Detects your Lossless Scaling DLL installation automatically.
*   **Easy Uninstallation:** Removes all plugin-related files when no longer needed.

## How to Get Started

1.  **Purchase and Install Lossless Scaling:** Get the original app from the [Steam Store](https://store.steampowered.com/app/993090/Lossless_Scaling/).
2.  **Download the Plugin:** Download the "Lossless Scaling.zip" file from the [latest release](https://github.com/xXJSONDeruloXx/decky-lossless-scaling-vk/releases) and save it on your Steam Deck.
3.  **Install via Decky Loader:**
    *   Enter Game Mode.
    *   Access Decky Loader settings (cog icon).
    *   Enable "Developer Mode".
    *   Go to the "Developer" tab and select "Install Plugin from Zip".
    *   Choose the downloaded "Lossless Scaling.zip" file.
4.  **Configure and Play:**
    *   Open the plugin from the Decky menu.
    *   Click "Install lsfg-vk".
    *   Adjust settings (FPS multiplier, flow scale, etc.).
    *   Add `~/lsfg %command%` to your game's launch options (or use the "Launch Option Clipboard").
    *   Launch your game and enjoy enhanced performance!

## Configuration Options

### Core Settings

*   **FPS Multiplier:** 2x, 3x, or 4x frame generation.
*   **Flow Scale:** Adjust motion estimation quality vs. performance.
*   **Performance Mode:** Optimized for most games.
*   **HDR Mode:** Enable for HDR-compatible games.

### Experimental Features

*   **Present Mode Override:** Force specific Vulkan presentation modes.
*   **Base FPS Limit:** Cap framerate before the multiplier.

## Troubleshooting

*   **Frame Generation Not Working:** Ensure the launch option is correct, the Lossless Scaling DLL is detected, and the game is in fullscreen mode.
*   **Performance Issues:** Lower the Flow Scale, enable Performance Mode, or reduce the FPS multiplier. Consider using the experimental FPS limit.

## Feedback and Support

For community support, please join the [Decky Lossless Discord Channel](https://discord.gg/TwvHdVucC3)

## Credits

*   **[PancakeTAS](https://github.com/PancakeTAS/lsfg-vk)** for creating the lsfg-vk Vulkan compatibility layer
*   **[Lossless Scaling](https://store.steampowered.com/app/993090/Lossless_Scaling/)** developers for the original frame generation technology
*   **[Deck Wizard](https://www.youtube.com/@DeckWizard)** for the helpful video tutorial
*   The **Decky Loader** team for the plugin framework
*   Community contributors and testers for feedback and bug reports