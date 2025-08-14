# NUR: The Nix User Repository

**Expand your Nix package options with the Nix User Repository (NUR), a community-driven platform for discovering and sharing Nix packages outside of Nixpkgs.** [Get started with NUR](https://github.com/nix-community/NUR)!

*   **Community-Driven:** Explore packages contributed and maintained by the Nix community.
*   **Faster Package Updates:** Access new packages and updates more quickly than through Nixpkgs.
*   **Decentralized:** Benefit from a more open and flexible package ecosystem.
*   **Easy Integration:** Integrate with flakes, `packageOverrides`, NixOS, Home Manager, and devshells.
*   **Comprehensive:** Find packages, modules, overlays and library functions.

## Table of Contents

*   [Installation](#installation)
    *   [Using Flakes](#using-flakes)
    *   [Using `packageOverrides`](#using-packageoverrides)
    *   [Pinning](#pinning)
*   [How to Use](#how-to-use)
    *   [Using a single package in a devshell](#using-a-single-package-in-a-devshell)
    *   [Using the flake in NixOS](#using-the-flake-in-nixos)
    *   [Integrating with Home Manager](#integrating-with-home-manager)
*   [Finding Packages](#finding-packages)
*   [How to Add Your Own Repository](#how-to-add-your-own-repository)
    *   [Use a different nix file as root expression](#use-a-different-nix-file-as-root-expression)
    *   [Update NUR's lock file after updating your repository](#update-nurs-lock-file-after-updating-your-repository)
    *   [Why are my NUR packages not updating?](#why-are-my-nur-packages-not-updating)
    *   [Git submodules](#git-submodules)
    *   [NixOS modules, overlays and library function support](#nixos-modules-overlays-and-library-function-support)
        *   [Providing NixOS modules](#providing-nixos-modules)
        *   [Providing Overlays](#providing-overlays)
        *   [Providing library functions](#providing-library-functions)
*   [Overriding Repositories](#overriding-repositories)
    *   [Overriding repositories with Flake](#overriding-repositories-with-flake)
*   [Contribution Guidelines](#contribution-guidelines)
*   [Contact](#contact)

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

Then use the overlay (`overlays.default`) or `legacyPackages.<system>`.

### Using `packageOverrides`

Add NUR to your `packageOverrides` in `~/.config/nixpkgs/config.nix`:

```nix
{
  packageOverrides = pkgs: {
    nur = import (builtins.fetchTarball "https://github.com/nix-community/NUR/archive/main.tar.gz") {
      inherit pkgs;
    };
  };
}
```

For NixOS, add it to `/etc/nixos/configuration.nix`.  Note: You may also need to include it in `~/.config/nixpkgs/config.nix` if you are using NUR with `nix-env`, `home-manager`, or `nix-shell`.

### Pinning

Pin the NUR version for more reliable builds:

```nix
builtins.fetchTarball {
  url = "https://github.com/nix-community/NUR/archive/3a6a6f4da737da41e27922ce2cfacf68a109ebce.tar.gz";
  sha256 = "04387gzgl8y555b3lkz9aiw9xsldfg4zmzp930m62qw8zbrvrshd";
}
```

## How to Use

Use packages from NUR:

```bash
$ nix-shell -p nur.repos.mic92.hello-nur
nix-shell> hello
Hello, NUR!
```

or

```bash
$ nix-env -f '<nixpkgs>' -iA nur.repos.mic92.hello-nur
```

or in `configuration.nix`:

```nix
environment.systemPackages = with pkgs; [
  nur.repos.mic92.hello-nur
];
```

**Important:** *Always review packages before installation as NUR packages are not reviewed.*

### Using a single package in a devshell

Here is an example of how to add a package from NUR in your `flake.nix`:

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

### Using the flake in NixOS

Integrate NUR modules and overlays in your NixOS configuration:

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
      # ...
      modules = [
        # Adds the NUR overlay
        nur.modules.nixos.default
        # NUR modules to import
        nur.legacyPackages."${system}".repos.iopq.modules.xraya
        # This adds the NUR nixpkgs overlay.
        # Example:
        # ({ pkgs, ... }: {
        #   environment.systemPackages = [ pkgs.nur.repos.mic92.hello-nur ];
        # })
      ];
    };
  };
}
```

### Integrating with Home Manager

Use NUR modules within your Home Manager configuration:

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

Browse packages via [Packages search for NUR](https://nur.nix-community.org/) or search in the [nur-combined](https://github.com/nix-community/nur-combined) repository.

## How to Add Your Own Repository

1.  Create a Git repository with a `default.nix` file. Use the [repository template](https://github.com/nix-community/nur-packages-template).
2.  Do not import packages from `<nixpkgs>`.  Instead, use the `pkgs` argument passed to your `default.nix`.

    ```nix
    { pkgs }:
    {
      hello-nur = pkgs.callPackage ./hello-nur {};
    }
    ```

    ```nix
    # ./hello-nur/default.nix
    { stdenv, fetchurl, lib }:

    stdenv.mkDerivation rec {
      pname = "hello";
      version = "2.10";

      src = fetchurl {
        url = "mirror://gnu/hello/${pname}-${version}.tar.gz";
        sha256 = "0ssi1wpaf7plaswqqjwigppsg5fyh99vdlb9kzl7c9lng89ndq1i";
      };

      postPatch = ''
        sed -i -e 's/Hello, world!/Hello, NUR!/' src/hello.c
      '';

      # fails due to patch
      doCheck = false;

      meta = with lib; {
        description = "A program that produces a familiar, friendly greeting";
        longDescription = ''
          GNU Hello is a program that prints "Hello, world!" when you run it.
          It is fully customizable.
        '';
        homepage = https://www.gnu.org/software/hello/manual/;
        changelog = "https://git.savannah.gnu.org/cgit/hello.git/plain/NEWS?h=v${version}";
        license = licenses.gpl3Plus;
        maintainers = [ maintainers.eelco ];
        platforms = platforms.all;
      };
    }
    ```

3.  Test your packages with `nix-shell` or `nix-build`.

    ```bash
    $ nix-shell --arg pkgs 'import <nixpkgs> {}' -A hello-nur
    nix-shell> hello
    Hello, NUR!
    ```

    ```bash
    $ nix-build --arg pkgs 'import <nixpkgs> {}' -A hello-nur
    ```

4.  (Optional) Set a default for the `pkgs` argument for easier development.

    ```nix
    { pkgs ? import <nixpkgs> {} }:
    {
      hello-nur = pkgs.callPackage ./hello-nur {};
    }
    ```
    ```bash
    $ nix-build -A hello-nur
    ```
5.  Add your repository to `repos.json` in the NUR repository:

    ```bash
    $ git clone --depth 1 https://github.com/nix-community/NUR
    $ cd NUR
    ```

    Edit `repos.json`:

    ```json
    {
        "repos": {
            "mic92": {
                "url": "https://github.com/Mic92/nur-packages"
            },
            "<fill-your-repo-name>": {
                "url": "https://github.com/<your-user>/<your-repo>"
            }
        }
    }
    ```

    Run `./bin/nur format-manifest` to sort `repos.json` alphabetically.

    Commit changes to `repos.json`, but *not* `repos.json.lock`.

    ```bash
    $ git add repos.json
    $ git commit -m "add <your-repo-name> repository"
    $ git push
    ```

6.  Open a pull request to [https://github.com/nix-community/NUR](https://github.com/nix-community/NUR).

### Use a different nix file as root expression

Use the `file` option to specify a different entry point.

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

### Update NUR's lock file after updating your repository

Use the update service after pushing changes to your repository: https://nur-update.nix-community.org/update?repo=mic92

Check out the [github page](https://github.com/nix-community/nur-update#nur-update-endpoint) for further details

### Why are my NUR packages not updating?

Check the [latest build job](https://github.com/nix-community/NUR/actions) for evaluation errors, such as:

*   Incorrect license attributes in the metadata.
*   Use `pkgs.fetch*` instead of `builtins.fetch*`.

#### Local evaluation check

Run the [check evaluation](https://github.com/nix-community/nur-packages-template/blob/main/.github/workflows/build.yml) task in your `nur-packages/` folder

```sh
nix-env -f . -qa \* --meta \
  --allowed-uris https://static.rust-lang.org \
  --option restrict-eval true \
  --option allow-import-from-derivation true \
  --drv-path --show-trace \
  -I nixpkgs=$(nix-instantiate --find-file nixpkgs) \
  -I ./ \
  --json | jq -r 'values | .[].name'
```

### Git submodules

Enable submodules using the `submodules` option:

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

### NixOS modules, overlays and library function support

Organize modules, overlays, and library functions within your repository for discoverability.

#### Providing NixOS modules

Place NixOS modules in the `modules` attribute:

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

Modules should be defined as paths, not functions, to avoid conflicts.

A module with no `_class` will be assumed to be both a NixOS and Home Manager module.  Use the `_class` attribute to specify `"nixos"` or `"home-manager"`.

#### Providing Overlays

Use the `overlays` attribute:

```nix
# default.nix
{
  overlays = {
    hello-overlay = import ./hello-overlay;
  };
}
```

```nix
# hello-overlay/default.nix
self: super: {
  hello = super.hello.overrideAttrs (old: {
    separateDebugInfo = true;
  });
}
```

#### Providing library functions

Put reusable Nix functions in the `lib` attribute:

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

Override repositories via the `repoOverrides` argument:

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

The repo must be a valid package repo.

### Overriding repositories with Flake

**Experimental**. You can override repositories in two ways:

-   With packageOverrides

```nix
{
  inputs.nur.url = "github:nix-community/NUR";
  inputs.paul.url = "path:/some_path/nur-paul"; # example: a local nur.repos.paul for development

  outputs = {self, nixpkgs, nur, paul }: {

  system = "x86_64-linux";

  nurpkgs = import nixpkgs { inherit system; };

  ...
  modules = [
       {
         nixpkgs.config.packageOverrides = pkgs: {
            nur = import nur {
              inherit pkgs nurpkgs;
              repoOverrides = { paul = import paul { inherit pkgs; }; };
            };
          };
        }
  ];
  ...
}
```

-   With overlay

```nix
{
  modules = [
    {
      nixpkgs.overlays = [
        (final: prev: {
          nur = import nur {
            nurpkgs = prev;
            pkgs = prev;
            repoOverrides = { paul = import paul { pkgs = prev; }; };
          };
        })
      ];
    }
    ...
  ];
}
```

The repo must contain a `flake.nix` file in addition to a `default.nix`:  [flake.nix example](https://github.com/Mic92/nur-packages/blob/master/flake.nix)

## Contribution Guidelines

*   Build packages and set `meta.broken = true;` if they don't build.
*   Provide standard meta attributes.
*   Keep repositories slim.
*   Reuse Nixpkgs packages where possible.

**Examples for packages that could be in NUR:**

*   Packages for a small audience.
*   Pre-releases.
*   Old versions no longer in Nixpkgs.
*   Automatic package generation (PyPI, CPAN, etc.).
*   Software with opinionated patches.
*   Experiments.

## Contact

Join us on the matrix channel [#nur:nixos.org](https://matrix.to/#/#nur:nixos.org) and also on [https://discourse.nixos.org/](https://discourse.nixos.org/).