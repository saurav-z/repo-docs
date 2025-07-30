# VRM Add-on for Blender: Seamlessly Integrate VRM Features into Blender

[![CI status](https://github.com/saturday06/VRM-Addon-for-Blender/actions/workflows/test.yml/badge.svg?branch=main)](https://github.com/saturday06/VRM-Addon-for-Blender/actions) [![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)

**Enhance your Blender workflow by easily importing, exporting, and working with VRM models using the VRM Add-on for Blender.**  ([View on GitHub](https://github.com/saturday06/VRM-Addon-for-Blender))

## Key Features

*   **Import and Export VRM Models:** Seamlessly bring your VRM models into Blender and export your creations in the VRM format.
*   **VRM Humanoid Support:** Utilize VRM Humanoid features within Blender for advanced character rigging and animation.
*   **MToon Shader Integration:** Apply the popular MToon shader for anime-style rendering directly within Blender.
*   **PBR Material Creation:** Create realistic physics-based rendering using PBR materials.
*   **Animation Support:** Animate your VRM models with ease using the built-in animation tools.
*   **Scripting API:** Automate tasks and extend functionality with Python scripting.

## Installation

### For Blender 4.2 or later:

*   Install from the [🛠️**Blender Extensions Platform**](https://extensions.blender.org/add-ons/vrm).

### For Blender 2.93 to 4.1:

*   Install from the [🌐**Official Site**](https://vrm-addon-for-blender.info).

## Tutorials

Explore detailed tutorials to get you started:

*   [Installation](https://vrm-addon-for-blender.info/en/installation?locale_redirection)
    <br/>
    <img width="200" src="https://vrm-addon-for-blender.info/images/installation.gif">
*   [Create Simple VRM](https://vrm-addon-for-blender.info/en/create-simple-vrm-from-scratch?locale_redirection)
    <br/>
    <img width="200" src="https://vrm-addon-for-blender.info/images/simple.gif">
*   [Create Humanoid VRM](https://vrm-addon-for-blender.info/en/create-humanoid-vrm-from-scratch?locale_redirection)
    <br/>
    <img width="200" src="https://vrm-addon-for-blender.info/images/humanoid.gif">
*   [Create Physics Based Material](https://vrm-addon-for-blender.info/en/material-pbr?locale_redirection)
    <br/>
    <img width="200" src="https://vrm-addon-for-blender.info/images/material_pbr.gif">
*   [Create Anime Style Material](https://vrm-addon-for-blender.info/en/material-mtoon?locale_redirection)
    <br/>
    <img width="200" src="https://vrm-addon-for-blender.info/images/material_mtoon.gif">
*   [Automation with Python Scripts](https://vrm-addon-for-blender.info/en/scripting-api?locale_redirection)
    <br/>
    <img width="200" src="https://vrm-addon-for-blender.info/images/scripting_api.gif">
*   [VRM Animation](https://vrm-addon-for-blender.info/en/animation?locale_redirection)
    <br/>
    <img width="200" src="https://vrm-addon-for-blender.info/images/animation.gif">
*   [Development How-To](https://vrm-addon-for-blender.info/en/development?locale_redirection)
    <br/>
    <img width="200" src="https://vrm-addon-for-blender.info/images/animation.gif">

## Development

The source code is in the [`main`](https://github.com/saturday06/VRM-Addon-for-Blender/tree/main) branch, particularly within the [`src/io_scene_vrm`](https://github.com/saturday06/VRM-Addon-for-Blender/tree/main/src/io_scene_vrm) directory.

For efficient development, create a symbolic link to the `io_scene_vrm` folder within your Blender addons directory. For advanced development, utilize [astral.sh/uv](https://docs.astral.sh/uv/) for running tests.

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

---

## Contributing

Bug reports, feature requests, pull requests, and general contributions are highly encouraged and welcome.