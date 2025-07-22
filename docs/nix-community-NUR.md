# Nix User Repository (NUR): Your Gateway to Community-Driven Nix Packages

NUR is a community-driven meta repository that expands the Nix ecosystem by providing access to user-contributed packages, offering a faster and more decentralized way to discover and install software not yet available in [Nixpkgs](https://github.com/NixOS/nixpkgs/).

*   **Community-Driven:** Discover packages built and maintained by the Nix community.
*   **Decentralized:** Access user-contributed packages without strict Nixpkgs review.
*   **Flexible:**  Easily install packages using flakes, `packageOverrides`, or integrate with NixOS configurations.
*   **Up-to-Date:** Automatic checks ensure repositories are evaluated and updated regularly.
*   **Extensible:**  Add your own packages and repositories with ease.

## Installation

### Using Flakes

Add NUR to your `flake.nix`:

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

Add NUR to your `~/.config/nixpkgs/config.nix`:

```nix
{
  packageOverrides = pkgs: {
    nur = import (builtins.fetchTarball "https://github.com/nix-community/NUR/archive/main.tar.gz") {
      inherit pkgs;
    };
  };
}
```

For NixOS, add to your `/etc/nixos/configuration.nix`:

```nix
{
  nixpkgs.config.packageOverrides = pkgs: {
    nur = import (builtins.fetchTarball "https://github.com/nix-community/NUR/archive/main.tar.gz") {
      inherit pkgs;
    };
  };
}
```

### Pinning

For stable builds, pin the NUR version:

```nix
builtins.fetchTarball {
  url = "https://github.com/nix-community/NUR/archive/3a6a6f4da737da41e27922ce2cfacf68a109ebce.tar.gz";
  sha256 = "04387gzgl8y555b3lkz9aiw9xsldfg4zmzp930m62qw8zbrvrshd";
}
```

## How to Use

Install packages with `nix-shell`, `nix-env`, or in your NixOS configuration:

```console
$ nix-shell -p nur.repos.mic92.hello-nur
```

or

```console
$ nix-env -f '<nixpkgs>' -iA nur.repos.mic92.hello-nur
```

or

```nix
# configuration.nix
environment.systemPackages = with pkgs; [
  nur.repos.mic92.hello-nur
];
```

***Always review packages before installation, as NUR does not provide strict security checks.***

## Example: DevShell using a Single Package

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

## Example: NixOS Configuration

Use NUR modules and overlays in your `configuration.nix`:

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

Integrate with Home Manager by adding your modules to the `imports` attribute.

```nix
let
  nur-no-pkgs = import (builtins.fetchTarball "https://github.com/nix-community/NUR/archive/main.tar.gz") {};
in
{
  imports = lib.attrValues nur-no-pkgs.repos.moredhel.hmModules.rawModules;

  services.unison = { ... };
}
```

## Finding Packages

*   [Packages search for NUR](https://nur.nix-community.org/)
*   [nur-combined](https://github.com/nix-community/nur-combined/search) on GitHub

## Adding Your Own Repository

1.  Create a repository with a `default.nix` file. See the [repository template](https://github.com/nix-community/nur-packages-template).
2.  Do not import packages with `with import <nixpkgs> {}`. Use the `pkgs` argument.
3.  Each repository should return a set of Nix derivations.
4.  Add your repository details to `repos.json` in the NUR repository.
5.  Run `./bin/nur format-manifest`, then add, commit, and push your `repos.json`.
6.  Open a pull request.

### Using a different nix file as root expression

Set the `file` option in `repos.json`:

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

To update NUR faster, use the service at https://nur-update.nix-community.org/ after pushing an update to your repository, e.g.:

```console
curl -XPOST https://nur-update.nix-community.org/update?repo=mic92
```

### Common Evaluation Errors

*   Incorrect license attributes in the metadata.
*   Use `pkgs.fetch*` instead of `builtins.fetch*`.

Check the [latest build job](https://github.com/nix-community/NUR/actions) to view evaluation results.

## Local Evaluation Check

In your `nur-packages/` folder, run the [check evaluation](https://github.com/nix-community/nur-packages-template/blob/main/.github/workflows/build.yml) task

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

## Git Submodules

To fetch git submodules in repositories set `submodules`:

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

## NixOS Modules, Overlays, and Library Function Support

Organize modules, overlays, and library functions within your repository.

### Providing NixOS Modules

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

Modules without a `_class` are assumed to be both NixOS and Home Manager modules.  Use `"nixos"` or `"home-manager"` for module-specific classifications.

### Providing Overlays

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
self: super: { ... }
```

### Providing Library Functions

Put reusable Nix functions in the `lib` attribute.

```nix
{ pkgs }:
with pkgs.lib;
{
  lib = { ... };
}
```

## Overriding Repositories

You can override repositories with `repoOverrides` argument:

```nix
{
  packageOverrides = pkgs: {
    nur = import (builtins.fetchTarball "https://github.com/nix-community/NUR/archive/main.tar.gz") {
      inherit pkgs;
      repoOverrides = {
        mic92 = import ../nur-packages { inherit pkgs; };
      };
    };
  };
}
```

### Overriding Repositories with Flake (Experimental)

Override repositories with `packageOverrides` or by using `nixpkgs.overlays` in your `flake.nix`. The repo must contain a `flake.nix`.

```nix
{
  inputs.nur.url = "github:nix-community/NUR";
  inputs.paul.url = "path:/some_path/nur-paul"; # example: a local nur.repos.paul for development

  outputs = { self, nixpkgs, nur, paul }: { ... };

  modules = [ { ...
    nixpkgs.config.packageOverrides = pkgs: {
       nur = import nur {
         inherit pkgs nurpkgs;
         repoOverrides = { paul = import paul { inherit pkgs; }; };
       };
     };
   } ];
}
```
## Contribution Guidelines

*   Build packages and set `meta.broken = true` if they fail.
*   Follow the Nixpkgs manual for `meta` attributes.
*   Keep repositories lean to optimize downloads.
*   Reuse packages from Nixpkgs where possible.

## Use Cases

NUR is ideal for:

*   Niche packages
*   Pre-releases
*   Legacy package versions
*   Automated package generation
*   Software with opinionated patches
*   Experiments

## Contact

Join us on [#nur:nixos.org](https://matrix.to/#/#nur:nixos.org) or the [NixOS Discourse](https://discourse.nixos.org/).

[Back to Top](#top)