# NUR: The Nix User Repository

**Expand your Nix package options with NUR, a community-driven repository offering user-submitted packages, modules, and more.**

[![GitHub Repo stars](https://img.shields.io/github/stars/nix-community/NUR?style=social)](https://github.com/nix-community/NUR)

## What is NUR?

The Nix User Repository (NUR) is a community-driven platform for Nix package management, providing access to a wide array of user-contributed Nix packages, modules, overlays, and library functions.  Unlike Nixpkgs, NUR allows for faster, decentralized sharing of packages built from source, offering a dynamic and extensive collection beyond the core NixOS packages.  This empowers users to easily discover and install packages by referencing their attributes.

## Key Features

*   **Community-Driven:** Access packages and configurations contributed by the Nix community.
*   **Decentralized:** Faster and more flexible package sharing compared to Nixpkgs.
*   **Package Variety:** Find packages, modules, overlays, and library functions.
*   **Automated Updates:** Benefit from automatic checks and evaluation before updates propagate.
*   **Easy Integration:**  Integrate with your existing Nix workflows using Flakes, `packageOverrides`, or direct package references.
*   **NixOS Module Support:**  Discover and utilize NixOS modules, overlays, and library functions within NUR.

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

Add the following to `~/.config/nixpkgs/config.nix`:

```nix
{
  packageOverrides = pkgs: {
    nur = import (builtins.fetchTarball "https://github.com/nix-community/NUR/archive/main.tar.gz") {
      inherit pkgs;
    };
  };
}
```

For NixOS, add to `/etc/nixos/configuration.nix`.

### Pinning

To ensure reliable builds, pin the NUR version using `fetchTarball` with a specific revision and `sha256` hash:

```nix
builtins.fetchTarball {
  url = "https://github.com/nix-community/NUR/archive/3a6a6f4da737da41e27922ce2cfacf68a109ebce.tar.gz";
  sha256 = "04387gzgl8y555b3lkz9aiw9xsldfg4zmzp930m62qw8zbrvrshd";
}
```
*Get the latest revision and hash from the [NUR GitHub repository](https://github.com/nix-community/NUR).*

## How to Use

Install and use packages from the NUR namespace in your shell or system configuration:

```bash
$ nix-shell -p nur.repos.mic92.hello-nur
# or
$ nix-env -f '<nixpkgs>' -iA nur.repos.mic92.hello-nur
# or
environment.systemPackages = with pkgs; [ nur.repos.mic92.hello-nur ]; # in configuration.nix
```

**Important:**  Always review expressions before installing packages from NUR.

### Example: Using a Single Package in a Devshell

This example demonstrates adding a single package from NUR to a devshell defined in a flake.nix.

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

### Using NUR in NixOS

Integrate NUR modules and overlays into your NixOS configuration with ease.

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

### Integrating with Home Manager

Integrate Home Manager by importing NUR modules into your `imports` attribute.

```nix
let
  nur-no-pkgs = import (builtins.fetchTarball "https://github.com/nix-community/NUR/archive/main.tar.gz") {};
in
{
  imports = lib.attrValues nur-no-pkgs.repos.moredhel.hmModules.rawModules;
}
```

## Finding Packages

*   **[Packages search for NUR](https://nur.nix-community.org/)**: Explore packages.
*   **[nur-combined](https://github.com/nix-community/nur-combined/search)**: Search across all user expressions.

## Contributing Your Own Repository

1.  Create a Git repository with a `default.nix` file.  Use the [repository template](https://github.com/nix-community/nur-packages-template) as a starting point.
2.  In your `default.nix`, import dependencies from the `pkgs` argument.
3.  Define a set of Nix derivations.
4.  Add your repository to `repos.json` in the NUR repository.
5.  Run `./bin/nur format-manifest` and commit changes.
6.  Create a pull request to [https://github.com/nix-community/NUR](https://github.com/nix-community/NUR).

### Use a different nix file as root expression

To use a different file instead of `default.nix` to load packages from, set the `file`
option to a path relative to the repository root:

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

### Updating NUR's Lock File

Use the NUR update service after pushing changes:
```bash
curl -XPOST https://nur-update.nix-community.org/update?repo=<your-repo-name>
```
*Find out more at the [nur-update documentation](https://github.com/nix-community/nur-update#nur-update-endpoint).*

### Why are my NUR packages not updating?

Ensure your package builds without errors. Common issues:
* Incorrect `meta.license` attribute in the metadata.
* Using `builtins.fetchGit` instead of `pkgs.fetchgit` (use pkgs.fetch* for external URL access)

Verify the evaluation success by checking the [latest build job](https://github.com/nix-community/NUR/actions).
You can also perform a local evaluation check (See original README).

### Git submodules

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

### NixOS modules, overlays and library function support

To make NixOS modules, overlays and library functions more discoverable,
they must be put them in their own namespace within the repository.

#### Providing NixOS modules

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

An example can be found [here](https://github.com/Mic92/nur-packages/tree/master/modules).
Modules should be defined as paths, not functions, to avoid conflicts if imported from multiple locations.

A module with no [_class](https://nixos.org/manual/nixpkgs/stable/index.html#module-system-lib-evalModules-param-class) will be assumed to be both a NixOS and Home Manager module. If a module is NixOS or Home Manager specific, the `_class` attribute should be set to `"nixos"` or [`"home-manager"`](https://github.com/nix-community/home-manager/commit/26e72d85e6fbda36bf2266f1447215501ec376fd).

#### Providing Overlays

For overlays, use the `overlays` attribute:

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

Put reusable nix functions that are intend for public use in the `lib` attribute:

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

Use `repoOverrides` to test changes.

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

### Overriding repositories with Flake

**Experimental**

- With packageOverrides
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
- With overlay
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

*   Ensure packages build successfully and set `meta.broken = true` if not.
*   Supply standard [Nixpkgs meta attributes](https://nixos.org/nixpkgs/manual/#sec-standard-meta-attributes).
*   Keep repositories concise.
*   Leverage packages from Nixpkgs when possible.

## Examples of NUR Packages

*   Packages for a smaller audience.
*   Pre-releases.
*   Old versions of packages.
*   Automatically generated package sets (e.g., from PyPI or CPAN).
*   Software with opinionated patches.
*   Experiments.

## Contact

*   Matrix: [#nur:nixos.org](https://matrix.to/#/#nur:nixos.org)
*   Discourse: [https://discourse.nixos.org](https://discourse.nixos.org/)

[Back to Top](#top)