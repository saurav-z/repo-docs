# NUR: The Nix User Repository

**Expand your Nix ecosystem with NUR, a community-driven repository for user-contributed Nix packages.**

[View the original repository](https://github.com/nix-community/NUR)

NUR (Nix User Repository) empowers Nix users to share and install packages more rapidly and collaboratively than traditional Nixpkgs. It offers a decentralized approach to package management, allowing you to access a wider array of software and customizations directly from the community.

**Key Features:**

*   **Community-Driven:** Access packages contributed and maintained by the Nix community.
*   **Faster Updates:** Get access to new packages and updates more quickly than through Nixpkgs.
*   **Decentralized:** Easily integrate packages from various user repositories.
*   **Flexible:** Supports packages built from source and offers options for NixOS modules, overlays, and library functions.
*   **Flake & PackageOverrides Support:** Integrate NUR into your Nix configurations using flakes or `packageOverrides`.
*   **Automated Evaluation Checks:** Ensure repository integrity with automatic checks before updates.
*   **Easy Package Discovery:** Find packages via the [Packages search for NUR](https://nur.nix-community.org/) or the [nur-combined](https://github.com/nix-community/nur-combined) repository.

## Installation

### Using Flakes

Integrate NUR into your `flake.nix`:

```nix
{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    nur = {
      url = "github:nix-community/NUR";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };
};
```

Then, use either the overlay (`overlays.default`) or `legacyPackages.<system>`.

### Using `packageOverrides`

Add this to `~/.config/nixpkgs/config.nix`:

```nix
{
  packageOverrides = pkgs: {
    nur = import (builtins.fetchTarball "https://github.com/nix-community/NUR/archive/main.tar.gz") {
      inherit pkgs;
    };
  };
}
```

For NixOS, add to `/etc/nixos/configuration.nix`. If using NUR in `nix-env`, `home-manager`, or `nix-shell`, also include it in `~/.config/nixpkgs/config.nix`.

### Pinning

For stability, pin the NUR version:

```nix
builtins.fetchTarball {
  url = "https://github.com/nix-community/NUR/archive/3a6a6f4da737da41e27922ce2cfacf68a109ebce.tar.gz";
  sha256 = "04387gzgl8y555b3lkz9aiw9xsldfg4zmzp930m62qw8zbrvrshd";
}
```

## How to Use

Use packages from the NUR namespace:

```bash
nix-shell -p nur.repos.mic92.hello-nur
```

or

```bash
nix-env -f '<nixpkgs>' -iA nur.repos.mic92.hello-nur
```

or in `configuration.nix`:

```nix
environment.systemPackages = with pkgs; [
  nur.repos.mic92.hello-nur
];
```

**Important:**  *Always review packages before installation, as they are not reviewed by Nixpkgs maintainers.*

## Example: Using a Single Package in a Devshell

```nix
{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    nur = {
      url = "github:nix-community/NUR";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, flake-utils, nur }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          overlays = [ nur.overlays.default ];
        };
      in
      {
        devShells.default = pkgs.mkShell {
          packages = [ pkgs.nur.repos.mic92.hello-nur ];
        };
      }
    );
}
```

## Using the Flake in NixOS

```nix
{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    nur = {
      url = "github:nix-community/NUR";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, nur }: {
    nixosConfigurations.myConfig = nixpkgs.lib.nixosSystem {
      modules = [
        nur.modules.nixos.default
        nur.legacyPackages."${system}".repos.iopq.modules.xraya
      ];
    };
  };
}
```

## Integrating with Home Manager

Add modules to `imports`:

```nix
let
  nur-no-pkgs = import (builtins.fetchTarball "https://github.com/nix-community/NUR/archive/main.tar.gz") {};
in
{
  imports = lib.attrValues nur-no-pkgs.repos.moredhel.hmModules.rawModules;

  services.unison = {
    enable = true;
    profiles = {
      org = {
        src = "/home/moredhel/org";
        dest = "/home/moredhel/org.backup";
        extraArgs = "-batch -watch -ui text -repeat 60 -fat";
      };
    };
  };
}
```

## Finding Packages

*   [Packages search for NUR](https://nur.nix-community.org/)
*   [nur-combined](https://github.com/nix-community/nur-combined)

## How to Contribute Your Own Repository

1.  Create a repository with a `default.nix` in the root.  Use the [repository template](https://github.com/nix-community/nur-packages-template).
2.  **Do not** import from `<nixpkgs>`.  Use the `pkgs` argument.
3.  Each repository should return a set of Nix derivations:

```nix
{ pkgs }:
{
  hello-nur = pkgs.callPackage ./hello-nur {};
}
```

4.  Add your repository to `repos.json` in NUR:

```bash
git clone --depth 1 https://github.com/nix-community/NUR
cd NUR
```

Edit `repos.json`:

```json
{
    "repos": {
        "mic92": {
            "url": "https://github.com/Mic92/nur-packages"
        },
        "<your-repo-name>": {
            "url": "https://github.com/<your-user>/<your-repo>"
        }
    }
}
```

5.  Run `./bin/nur format-manifest` and commit changes to `repos.json` (but *not* `repos.json.lock`).
6.  Open a pull request to [https://github.com/nix-community/NUR](https://github.com/nix-community/NUR).

### Root Expression Override

You can set the `file` option in `repos.json` to use a different nix file as the root expression:

```json
{
    "repos": {
        "mic92": {
            "url": "https://github.com/Mic92/nur-packages",
            "file": "subdirectory/default.nix"
        }
    }
}
```

### Update the NUR Lock File

Use the [nur-update service](https://nur-update.nix-community.org/) after updating your repository:

```bash
curl -XPOST https://nur-update.nix-community.org/update?repo=mic92
```

### Troubleshooting Package Updates

*   Check the [latest build job](https://github.com/nix-community/NUR/actions) for evaluation errors.
*   Common errors: incorrect licenses or using built-in fetchers.
*   Run the [check evaluation](https://github.com/nix-community/nur-packages-template/blob/main/.github/workflows/build.yml) task locally.

### Git Submodules

Enable submodules:

```json
{
    "repos": {
        "mic92": {
            "url": "https://github.com/Mic92/nur-packages",
            "submodules": true
        }
    }
}
```

### NixOS Modules, Overlays, and Library Functions

Structure them within your repository:

#### NixOS Modules

```nix
{ pkgs }: {
  modules = import ./modules;
}
```

```nix
# modules/default.nix
{
  example-module = ./example-module.nix;
}
```

#### Overlays

```nix
# default.nix
{
  overlays = {
    hello-overlay = import ./hello-overlay;
  };
}
```

#### Library Functions

```nix
{ pkgs }:
with pkgs.lib;
{
  lib = {
    hexint = x: hexvals.${toLower x};

    hexvals = listToAttrs (imap (i: c: { name = c; value = i - 1; })
      (stringToCharacters "0123456789abcdef"));
  };
}
```

## Overriding Repositories

Override repositories using the `repoOverrides` argument.

```nix
{
  packageOverrides = pkgs: {
    nur = import (builtins.fetchTarball "https://github.com/nix-community/NUR/archive/main.tar.gz") {
      inherit pkgs;
      repoOverrides = {
        mic92 = import ../nur-packages { inherit pkgs; };
        ## remote locations are also possible:
        # mic92 = import (builtins.fetchTarball "https://github.com/your-user/nur-packages/archive/main.tar.gz") { inherit pkgs; };
      };
    };
  };
}
```

or with Flakes.  See the original README for Flake implementation details.

## Contribution Guidelines

*   Ensure packages build and set `meta.broken` if not.
*   Supply standard `meta` attributes.
*   Keep repositories slim.
*   Reuse packages from Nixpkgs when applicable.

## Examples for Packages in NUR

*   Packages for a small audience
*   Pre-releases
*   Old package versions
*   Automatically generated package sets
*   Software with opinionated patches
*   Experiments

## Contact

*   Matrix: [#nur:nixos.org](https://matrix.to/#/#nur:nixos.org)
*   Discourse: [https://discourse.nixos.org/](https://discourse.nixos.org/)