# VRM Add-on for Blender: Create and Customize VRM Models Directly in Blender

Enhance your Blender workflow with the VRM Add-on, empowering you to import, export, and create VRM models seamlessly.  ([Back to the GitHub Repository](https://github.com/saturday06/VRM-Addon-for-Blender))

## Key Features:

*   **Import & Export VRM:** Easily load and save VRM models within Blender.
*   **VRM Humanoid Support:**  Simplify character rigging with built-in VRM humanoid functionality.
*   **MToon Shader Integration:**  Apply the popular MToon shader for anime-style rendering directly within Blender.
*   **PBR Material Support:** Create physically based rendering materials.
*   **Animation Support:** Create VRM animations.
*   **Scripting API:** Automate tasks and extend functionality with Python scripts.

## Getting Started

### Download

*   **Blender 4.2 or later:**  Download the add-on from the [🛠️Blender Extensions Platform](https://extensions.blender.org/add-ons/vrm).
*   **Blender 2.93 to 4.1:** Download from the [🌐Official Site](https://vrm-addon-for-blender.info).

## Tutorials

[Here are some handy tutorials to get you started.]

|                                         [Installation](https://vrm-addon-for-blender.info/en/installation?locale_redirection)                                          |                                    [Create Simple VRM](https://vrm-addon-for-blender.info/en/create-simple-vrm-from-scratch?locale_redirection)                                    |                                    [Create Humanoid VRM](https://vrm-addon-for-blender.info/en/create-humanoid-vrm-from-scratch?locale_redirection)                                    |
| :--------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| <a href="https://vrm-addon-for-blender.info/en/installation?locale_redirection"><img width="200" src="https://vrm-addon-for-blender.info/images/installation.gif"></a> | <a href="https://vrm-addon-for-blender.info/en/create-simple-vrm-from-scratch?locale_redirection"><img width="200" src="https://vrm-addon-for-blender.info/images/simple.gif"></a> | <a href="https://vrm-addon-for-blender.info/en/create-humanoid-vrm-from-scratch?locale_redirection"><img width="200" src="https://vrm-addon-for-blender.info/images/humanoid.gif"></a> |
|                               **[Create Physics Based Material](https://vrm-addon-for-blender.info/en/material-pbr?locale_redirection)**                               |                                     **[Create Anime Style Material](https://vrm-addon-for-blender.info/en/material-mtoon?locale_redirection)**                                     |                                      **[Automation with Python Scripts](https://vrm-addon-for-blender.info/en/scripting-api?locale_redirection)**                                      |
| <a href="https://vrm-addon-for-blender.info/en/material-pbr?locale_redirection"><img width="200" src="https://vrm-addon-for-blender.info/images/material_pbr.gif"></a> |     <a href="https://vrm-addon-for-blender.info/en/material-mtoon?locale_redirection"><img width="200" src="https://vrm-addon-for-blender.info/images/material_mtoon.gif"></a>     |        <a href="https://vrm-addon-for-blender.info/en/scripting-api?locale_redirection"><img width="200" src="https://vrm-addon-for-blender.info/images/scripting_api.gif"></a>        |
|                                        **[VRM Animation](https://vrm-addon-for-blender.info/en/animation?locale_redirection)**                                         |                                           **[Development How-To](https://vrm-addon-for-blender.info/en/development?locale_redirection)**                                           |                                                                                                                                                                                        |
|    <a href="https://vrm-addon-for-blender.info/en/animation?locale_redirection"><img width="200" src="https://vrm-addon-for-blender.info/images/animation.gif"></a>    |         <a href="https://vrm-addon-for-blender.info/en/development?locale_redirection"><img width="200" src="https://vrm-addon-for-blender.info/images/animation.gif"></a>         |                                                                                                                                                                                        |

## Development

The source code is located in the `main` branch. Detailed development setup instructions are in the [Development How-To](https://vrm-addon-for-blender.info/en/development?locale_redirection)

```text
git checkout main

# Blender 4.2 or later

# Linux
ln -Ts "$PWD/src/io_scene_vrm" "$HOME/.config/blender/BLENDER_VERSION/extensions/user_default/vrm"
# macOS
ln -s "$PWD/src/io_scene_vrm" "$HOME/Library/Application Support/Blender/BLENDER_VERSION/extensions/user_default/vrm"
# Windows PowerShell
New-Item -ItemType Junction -Path "$Env:APPDATA\Blender Foundation\Blender\BLENDER_VERSION\extensions\user_default\vrm" -Value "$(Get-Location)\src\io_scene_vrm"
# Windows Command Prompt
mklink /j "%APPDATA%\Blender Foundation\Blender\BLENDER_VERSION\extensions\user_default\vrm" src\io_scene_vrm

# Blender 4.1.1 or earlier

# Linux
ln -Ts "$PWD/src/io_scene_vrm" "$HOME/.config/blender/BLENDER_VERSION/scripts/addons/io_scene_vrm"
# macOS
ln -s "$PWD/src/io_scene_vrm" "$HOME/Library/Application Support/Blender/BLENDER_VERSION/scripts/addons/io_scene_vrm"
# Windows PowerShell
New-Item -ItemType Junction -Path "$Env:APPDATA\Blender Foundation\Blender\BLENDER_VERSION\scripts\addons\io_scene_vrm" -Value "$(Get-Location)\src\io_scene_vrm"
# Windows Command Prompt
mklink /j "%APPDATA%\Blender Foundation\Blender\BLENDER_VERSION\scripts\addons\io_scene_vrm" src\io_scene_vrm
```