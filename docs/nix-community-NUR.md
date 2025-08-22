# NUR: The Nix User Repository

**Expand your Nix package library with NUR, a community-driven repository for user-contributed packages.**

[Original Repo](https://github.com/nix-community/NUR)

NUR (Nix User Repository) empowers the Nix community by providing a decentralized platform for sharing and installing Nix packages. Unlike Nixpkgs, packages in NUR are built from source and are not subject to Nixpkgs review, enabling rapid access to cutting-edge software and community-created configurations.

## Key Features

*   **Community-Driven:** Access a vast collection of user-contributed packages.
*   **Decentralized:** Faster package availability without the constraints of a centralized review process.
*   **Flexible Installation:** Install packages through various methods, including flakes, `packageOverrides`, and direct integration with NixOS configurations and Home Manager.
*   **Automatic Updates:** NUR automatically checks repositories and performs evaluation checks, propagating updates.
*   **NixOS Modules, Overlays & Library Functions:**  Easily integrates NixOS modules, overlays, and library functions.
*   **Package Search:** Find packages using the [NUR Packages Search](https://nur.nix-community.org/) or the [nur-combined](https://github.com/nix-community/nur-combined) repository.
*   **Easy Contribution:** Add your own packages or configurations to NUR.

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

Then, either the overlay (`overlays.default`) or `legacyPackages.<system>` can be used.

### Using `packageOverrides`

Add the following to your `~/.config/nixpkgs/config.nix`:

```nix
{
  packageOverrides = pkgs: {
    nur = import (builtins.fetchTarball "https://github.com/nix-community/NUR/archive/main.tar.gz") {
      inherit pkgs;
    };
  };
}
```

For NixOS, add the following to your `/etc/nixos/configuration.nix`. If you want to use NUR in nix-env, home-manager or in nix-shell you also need NUR in `~/.config/nixpkgs/config.nix` as shown above!

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

Pinning the version is recommended to ensure build reproducibility.

```nix
builtins.fetchTarball {
  url = "https://github.com/nix-community/NUR/archive/3a6a6f4da737da41e27922ce2cfacf68a109ebce.tar.gz";
  sha256 = "04387gzgl8y555b3lkz9aiw9xsldfg4zmzp930m62qw8zbrvrshd";
}
```

## How to Use

Install and use packages from NUR directly in your Nix environments:

```console
$ nix-shell -p nur.repos.mic92.hello-nur
nix-shell> hello
Hello, NUR!
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

**Important:** *It is recommended to check expressions before installing.*

## Examples

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

*   **Packages Search for NUR:** [https://nur.nix-community.org/](https://nur.nix-community.org/)
*   **nur-combined (GitHub):** [https://github.com/nix-community/nur-combined/search](https://github.com/nix-community/nur-combined/search)

## Contributing Your Repository

1.  **Create a Repository:** Structure your repository with a `default.nix` file (template available).
2.  **Dependencies:** Import packages from Nixpkgs using the `pkgs` argument.
3.  **Build Packages:** Use `nix-shell` or `nix-build`.
4.  **Add to `repos.json`:**
    *   Clone the NUR repository.
    *   Edit `repos.json` to include your repository's URL.
    *   Run `./bin/nur format-manifest` and commit the changes.
    *   Open a pull request to the NUR repository.

**Note:** Repositories must be buildable on Nixpkgs unstable.

### Additional Configuration

*   **`file` Option:** Specify a different root expression file.
*   **`submodules` Option:** Enable Git submodule support.

### Updating NUR's Lock File

Use [https://nur-update.nix-community.org/](https://nur-update.nix-community.org/) to trigger an update after you've pushed changes to your repository.

### Troubleshooting

*   **Evaluation Errors:** Ensure your package expressions are valid and dependencies are correctly specified.
*   **Check Build Logs:** Review the [latest build job](https://github.com/nix-community/NUR/actions) for any errors.
*   **Local Evaluation Check:** Run the provided [check evaluation task](https://github.com/nix-community/nur-packages-template/blob/main/.github/workflows/build.yml) to validate your packages locally.

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

It is also possible to define more than just packages. In fact any Nix expression can be used.

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

You can override repositories using `repoOverrides` argument.
This allows to test changes before publishing.

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

*   **Build and Meta Attributes:** Ensure packages build and include essential `meta` attributes.
*   **Slim Repositories:** Keep your repositories concise.
*   **Reuse Nixpkgs:** Utilize packages from Nixpkgs when feasible.

## Contact

Join the conversation:

*   **Matrix Channel:** [#nur:nixos.org](https://matrix.to/#/#nur:nixos.org)
*   **Discourse:** [https://discourse.nixos.org/](https://discourse.nixos.org/)