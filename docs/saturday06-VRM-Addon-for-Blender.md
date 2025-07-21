# VRM Add-on for Blender: Create, Import, and Export VRM Models

[![CI status](https://github.com/saturday06/VRM-Addon-for-Blender/actions/workflows/test.yml/badge.svg?branch=main)](https://github.com/saturday06/VRM-Addon-for-Blender/actions) [![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)

**Enhance your Blender workflow with the VRM Add-on, enabling seamless VRM model creation, import, and export for your 3D projects.** This add-on brings essential VRM functionalities directly into Blender, allowing users to work with VRM models more easily.

[View the original repository](https://github.com/saturday06/VRM-Addon-for-Blender)

## Key Features

*   **Import & Export VRM:** Easily import and export VRM models within Blender.
*   **VRM Humanoid Support:** Integrate VRM Humanoid features for character rigging and animation.
*   **MToon Shader Integration:** Utilize MToon shaders for creating visually appealing anime-style materials.
*   **Physics Based Material Support:** Support for PBR materials.
*   **Scripting API:** Automate tasks with Python scripts.

## Download

*   **For Blender 4.2 or later:**
    [🛠️**Blender Extensions Platform**](https://extensions.blender.org/add-ons/vrm)
*   **For Blender 2.93 to 4.1:**
    [🌐**The Official Site**](https://vrm-addon-for-blender.info)

## Tutorials

| Installation                                         | Create Simple VRM                                       | Create Humanoid VRM                                      |
| :----------------------------------------------------: | :-----------------------------------------------------: | :------------------------------------------------------: |
| <a href="https://vrm-addon-for-blender.info/en/installation?locale_redirection"><img width="200" src="https://vrm-addon-for-blender.info/images/installation.gif"></a> | <a href="https://vrm-addon-for-blender.info/en/create-simple-vrm-from-scratch?locale_redirection"><img width="200" src="https://vrm-addon-for-blender.info/images/simple.gif"></a> | <a href="https://vrm-addon-for-blender.info/en/create-humanoid-vrm-from-scratch?locale_redirection"><img width="200" src="https://vrm-addon-for-blender.info/images/humanoid.gif"></a> |
| Create Physics Based Material                           | Create Anime Style Material                              | Automation with Python Scripts                             |
| <a href="https://vrm-addon-for-blender.info/en/material-pbr?locale_redirection"><img width="200" src="https://vrm-addon-for-blender.info/images/material_pbr.gif"></a> | <a href="https://vrm-addon-for-blender.info/en/material-mtoon?locale_redirection"><img width="200" src="https://vrm-addon-for-blender.info/images/material_mtoon.gif"></a> | <a href="https://vrm-addon-for-blender.info/en/scripting-api?locale_redirection"><img width="200" src="https://vrm-addon-for-blender.info/images/scripting_api.gif"></a> |
| VRM Animation                                         | Development How-To                                     |                                                          |
| <a href="https://vrm-addon-for-blender.info/en/animation?locale_redirection"><img width="200" src="https://vrm-addon-for-blender.info/images/animation.gif"></a> | <a href="https://vrm-addon-for-blender.info/en/development?locale_redirection"><img width="200" src="https://vrm-addon-for-blender.info/images/animation.gif"></a> |                                                          |

## Development

The source code for development is located in the `main` branch and the `src/io_scene_vrm` folder. For more advanced development, please use [astral.sh/uv](https://docs.astral.sh/uv/).

For development setup instructions, see:
[the tutorial](https://vrm-addon-for-blender.info/en/development?locale_redirection)
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