# Nix User Repository (NUR): Community-Driven Package Management for Nix

**Expand your Nix ecosystem with the Nix User Repository (NUR), a community-driven platform offering a vast collection of user-contributed packages.** [Learn more at the original repo](https://github.com/nix-community/NUR).

NUR allows you to access and install packages created and maintained by the Nix community, offering a more decentralized and rapid approach to sharing and discovering new software. Unlike Nixpkgs, NUR packages are built from source and are not reviewed by Nixpkgs maintainers, enabling faster access to cutting-edge software and specialized packages.

**Key Features:**

*   **Community-Driven:** Access a wide array of packages contributed by Nix users.
*   **Rapid Package Availability:** Get access to new packages quickly, bypassing the standard Nixpkgs review process.
*   **Decentralized:** Leverage a distributed system for package management and sharing.
*   **Automatic Updates:**  NUR automatically checks and evaluates updates, ensuring package integrity.
*   **Flexible Installation:** Integrate NUR packages through flakes, package overrides, and direct usage.
*   **Extensive Package Search:** Discover packages through a dedicated web interface and the `nur-combined` repository.
*   **Support for NixOS Modules, Overlays, and Library Functions:** Extend your Nix configurations with custom modules and overlays from community repositories.

## Installation

### Using Flakes

Integrate NUR into your `flake.nix` with:

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

Add NUR to your `packageOverrides`:

```nix
{
  packageOverrides = pkgs: {
    nur = import (builtins.fetchTarball "https://github.com/nix-community/NUR/archive/main.tar.gz") {
      inherit pkgs;
    };
  };
}
```

Update `~/.config/nixpkgs/config.nix` for user-level access and `/etc/nixos/configuration.nix` for NixOS. Remember, you will also need NUR in `~/.config/nixpkgs/config.nix` if using NUR with `nix-env`, `home-manager` or in `nix-shell`.

### Pinning

Pin specific NUR versions for consistent builds:

```nix
builtins.fetchTarball {
  url = "https://github.com/nix-community/NUR/archive/3a6a6f4da737da41e27922ce2cfacf68a109ebce.tar.gz";
  sha256 = "04387gzgl8y555b3lkz9aiw9xsldfg4zmzp930m62qw8zbrvrshd";
}
```

## How to Use

Install and use NUR packages seamlessly:

```console
$ nix-shell -p nur.repos.mic92.hello-nur
nix-shell> hello
Hello, NUR!
```

Or with `nix-env`:

```console
$ nix-env -f '<nixpkgs>' -iA nur.repos.mic92.hello-nur
```

And in `configuration.nix`:

```nix
environment.systemPackages = with pkgs; [
  nur.repos.mic92.hello-nur
];
```

**Important:** *Always review packages before installation.*

### Example: Using a Single Package in a Devshell

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

### Using the Flake in NixOS

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
*   [nur-combined](https://github.com/nix-community/nur-combined/search)

## Contributing Your Own Repository

1.  Create a Git repository with a `default.nix` at the root.
2.  Structure your repository to export Nix derivations.
3.  **Important:**  Use the `pkgs` argument provided to access packages from Nixpkgs (e.g., `pkgs.hello`).
4.  Test your packages with `nix-build` and `nix-shell`.
5.  Add your repository details to `repos.json` (e.g., `"your_repo_name": { "url": "your_repo_url" }`).
6.  Run `./bin/nur format-manifest` to sort the `repos.json`.
7.  Commit and push your changes.
8.  Submit a pull request to the NUR repository.

### Use a Different Nix File as Root Expression

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

### Update NUR's Lock File

Use the NUR update service to refresh faster:
```console
curl -XPOST https://nur-update.nix-community.org/update?repo=mic92
```

### Resolving Package Update Issues

Ensure your packages evaluate successfully and resolve common issues:

*   Verify you are using correct metadata attributes.
*   Use `pkgs.fetch*` functions instead of `builtins.fetch*`.

You can find out if your evaluation succeeded by checking the [latest build job](https://github.com/nix-community/NUR/actions).

#### Local evaluation check

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

On success, this shows a list of your packages

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

### NixOS Modules, Overlays, and Library Function Support

Organize your NixOS modules, overlays, and library functions into the appropriate namespaces within your repository.

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

Use the `repoOverrides` argument to test changes before publishing:

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

The repo must be a valid package repo, i.e. its root contains a `default.nix` file.

### Overriding repositories with Flake

**Experimental** Note that flake support is still experimental and might change in future in a backwards incompatible way.

You can override repositories in two ways:

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

*   Ensure packages build and set `meta.broken = true;` if not.
*   Use standard Nixpkgs meta attributes.
*   Keep repositories lean for efficiency.
*   Reuse packages from Nixpkgs when possible.

## Examples of Suitable NUR Packages

*   Packages for a small audience
*   Pre-release software
*   Legacy package versions
*   Automated package sets (e.g., from PyPI or CPAN)
*   Software with specific patches
*   Experiments

## Contact

*   Matrix: [#nur:nixos.org](https://matrix.to/#/#nur:nixos.org)
*   Discourse: [https://discourse.nixos.org/](https://discourse.nixos.org/)