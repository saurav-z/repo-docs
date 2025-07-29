# VRM Add-on for Blender: Create, Import, and Export VRM Models

[![CI status](https://github.com/saturday06/VRM-Addon-for-Blender/actions/workflows/test.yml/badge.svg?branch=main)](https://github.com/saturday06/VRM-Addon-for-Blender/actions)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://docs.astral.sh/uv)

**Enhance your Blender workflow by seamlessly importing, exporting, and manipulating VRM models with the VRM Add-on for Blender!** This add-on empowers Blender users to create, edit, and optimize VRM models for various applications, including virtual avatars, game development, and more.

**[View the original repository](https://github.com/saturday06/VRM-Addon-for-Blender)**

## Key Features

*   **Import & Export:** Import and export VRM files directly within Blender.
*   **VRM Humanoid Support:** Easily add and configure VRM Humanoid rigs for character animation.
*   **MToon Shader Support:** Implement the popular MToon shader for anime-style rendering.
*   **Physics-Based Materials:** Create physically accurate materials for realistic rendering.
*   **Animation Tools:** Animate your VRM models directly within Blender.
*   **Scripting API:** Automate tasks and extend functionality using Python scripts.

## Download

Choose the correct download based on your Blender version:

*   **Blender 4.2 and later:** Available on the [🛠️**Blender Extensions Platform**](https://extensions.blender.org/add-ons/vrm).
*   **Blender 2.93 to 4.1:** Download from [🌐**The Official Site**](https://vrm-addon-for-blender.info).

## Tutorials

Explore detailed tutorials to get you started:

|                                         Installation                                         |                                    Create Simple VRM                                    |                                   Create Humanoid VRM                                   |
| :-----------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------: |
| <a href="https://vrm-addon-for-blender.info/en/installation?locale_redirection"><img width="200" src="https://vrm-addon-for-blender.info/images/installation.gif"></a> | <a href="https://vrm-addon-for-blender.info/en/create-simple-vrm-from-scratch?locale_redirection"><img width="200" src="https://vrm-addon-for-blender.info/images/simple.gif"></a> | <a href="https://vrm-addon-for-blender.info/en/create-humanoid-vrm-from-scratch?locale_redirection"><img width="200" src="https://vrm-addon-for-blender.info/images/humanoid.gif"></a> |
|                                 Create Physics Based Material                                 |                                   Create Anime Style Material                                   |                                    Automation with Python Scripts                                     |
| <a href="https://vrm-addon-for-blender.info/en/material-pbr?locale_redirection"><img width="200" src="https://vrm-addon-for-blender.info/images/material_pbr.gif"></a> |  <a href="https://vrm-addon-for-blender.info/en/material-mtoon?locale_redirection"><img width="200" src="https://vrm-addon-for-blender.info/images/material_mtoon.gif"></a> | <a href="https://vrm-addon-for-blender.info/en/scripting-api?locale_redirection"><img width="200" src="https://vrm-addon-for-blender.info/images/scripting_api.gif"></a> |
|                                          VRM Animation                                          |                                         Development How-To                                         |                                                                                                   |
|  <a href="https://vrm-addon-for-blender.info/en/animation?locale_redirection"><img width="200" src="https://vrm-addon-for-blender.info/images/animation.gif"></a> | <a href="https://vrm-addon-for-blender.info/en/development?locale_redirection"><img width="200" src="https://vrm-addon-for-blender.info/images/animation.gif"></a> |                                                                                                   |

## Development

The source code is located in the `main` branch and the main body of the add-on is in the `src/io_scene_vrm` folder.  For efficient development, link this folder into Blender's addon folder.

For advanced development tasks, such as running tests, leverage [astral.sh/uv](https://docs.astral.sh/uv/). Refer to the [tutorial](https://vrm-addon-for-blender.info/en/development?locale_redirection) for detailed instructions.

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