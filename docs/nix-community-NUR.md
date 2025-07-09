# Nix User Repository (NUR): Access Community-Driven Nix Packages

**NUR provides a decentralized way to access and install community-contributed Nix packages, offering a broader range of software than Nixpkgs.** Find out more at the [original repo](https://github.com/nix-community/NUR).

**Key Features:**

*   **Community-Driven:** Access packages created and maintained by the Nix community.
*   **Decentralized:**  Share and install packages faster, without requiring strict Nixpkgs review.
*   **Flake and Package Override Support:**  Easily integrate NUR packages into your Nix configurations using flakes or package overrides.
*   **Automatic Evaluation Checks:** NUR automatically checks repositories for errors and updates.
*   **NixOS Module, Overlay and Library Function Support:** Define and discover NixOS modules, overlays, and reusable functions.
*   **Easy Package Discovery:** Find packages through the NUR package search or the nur-combined repository.
*   **Repository Updates:**  Use the nur-update service to ensure your repository is current.
*   **Customizable Repositories:** Add your own repository with a simple `default.nix` file and register it in NUR.

## Installation

### Using Flakes

Include NUR in your `flake.nix`:

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

Then, use the overlay (`overlays.default`) or `legacyPackages.<system>`.

### Using `packageOverrides`

Add the following to `~/.config/nixpkgs/config.nix` or `/etc/nixos/configuration.nix` for NixOS:

```nix
{
  packageOverrides = pkgs: {
    nur = import (builtins.fetchTarball "https://github.com/nix-community/NUR/archive/main.tar.gz") {
      inherit pkgs;
    };
  };
}
```

### Pinning

Pin the version to avoid caching issues:

```nix
builtins.fetchTarball {
  url = "https://github.com/nix-community/NUR/archive/3a6a6f4da737da41e27922ce2cfacf68a109ebce.tar.gz";
  sha256 = "04387gzgl8y555b3lkz9aiw9xsldfg4zmzp930m62qw8zbrvrshd";
}
```

## How to Use

Install packages using `nix-shell`, `nix-env`, or by including them in your `configuration.nix`:

```console
$ nix-shell -p nur.repos.mic92.hello-nur
nix-shell> hello
```

or

```console
$ nix-env -f '<nixpkgs>' -iA nur.repos.mic92.hello-nur
```

or

```console
# configuration.nix
environment.systemPackages = with pkgs; [
  nur.repos.mic92.hello-nur
];
```

***Important Security Note:** Carefully review packages before installation, as NUR packages are not reviewed by Nixpkgs maintainers.*

### Using a single package in a devshell

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
          overlays = [ nur.overlay ];
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
*   [nur-combined](https://github.com/nix-community/nur-combined/search) repository

## Adding Your Own Repository

1.  Create a repository with a `default.nix` at the top level (use the [template](https://github.com/nix-community/nur-packages-template)).
2.  Do NOT import packages directly from `<nixpkgs>`. Instead, take all dependencies you want to import from Nixpkgs from the given `pkgs` argument.
3.  Your `default.nix` should return a set of Nix derivations.

    ```nix
    { pkgs }:
    {
      hello-nur = pkgs.callPackage ./hello-nur {};
    }
    ```

    Where `hello-nur` is a directory containing a `default.nix`:

    ```nix
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
4.  Test your packages with `nix-shell` or `nix-build`.
5.  Add your repository to `repos.json` in the NUR repository:

    ```console
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

    *Note: The URL must point to a Git repository.*
6.  Run `./bin/nur update` and test.
7.  Commit and push the changed `repos.json` (but not `repos.json.lock`).
8.  Open a pull request at [https://github.com/nix-community/NUR](https://github.com/nix-community/NUR).

### Using a different nix file as root expression

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

Use the nur-update service to update NUR faster:

```console
curl -XPOST https://nur-update.nix-community.org/update?repo=mic92
```

### Why are my NUR packages not updating?

*   Make sure your evaluation doesn't have any errors.
*   Common errors are:
    *   Incorrect license attribute
    *   Using builtins fetchers
*   Check the [latest build job](https://github.com/nix-community/NUR/actions) to check if your evaluation succeeded

#### Local evaluation check

Run the [check evaluation](https://github.com/nix-community/nur-packages-template/blob/main/.github/workflows/build.yml) task to your repository

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

NixOS modules should be placed in the `modules` attribute:

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

A module with no [_class](https://nixos.org/manual/nixpkgs/stable/index.html#module-system-lib-evalModules-param-class) will be assumed to be both a NixOS and Home Manager module. If a module is NixOS or Home Manager specific, the `_class` attribute should be set to `"nixos"` or [`"home-manager"`](https://github.com/nix-community/home-manager/commit/26e72d85e6fbda36bf2266f1447215501ec376fd).

Overlays should go in the `overlays` attribute:

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

Library functions should go in the `lib` attribute:

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

Use the `repoOverrides` argument to test changes:

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

### Overriding repositories with Flake

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

Or, with overlay:

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

## Contribution Guidelines

*   Ensure packages build and set `meta.broken = true` if they do not.
*   Use standard [Nixpkgs meta attributes](https://nixos.org/nixpkgs/manual/#sec-standard-meta-attributes) for discoverability.
*   Keep repositories small.
*   Reuse packages from Nixpkgs when possible.

## Contact

*   Matrix: [#nur:nixos.org](https://matrix.to/#/#nur:nixos.org)
*   Discourse: [https://discourse.nixos.org](https://discourse.nixos.org/)