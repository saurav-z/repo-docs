# Nix User Repository (NUR): Expand Your Nix Package Ecosystem

NUR is a community-driven meta-repository that allows you to easily discover and install user-contributed Nix packages, offering a faster and more decentralized way to share and use software. [Explore NUR on GitHub](https://github.com/nix-community/NUR).

**Key Features:**

*   **Community-Driven:** Access packages contributed and maintained by the Nix community.
*   **Decentralized Package Sharing:** Quickly share and install packages not yet available in Nixpkgs.
*   **Automatic Updates & Checks:** NUR automatically checks and validates repositories for updates and evaluation.
*   **Flexible Installation:** Install packages using flakes, `packageOverrides`, or directly in your NixOS configuration.
*   **NixOS Modules & Overlays:** Supports NixOS modules, overlays, and library functions for advanced configuration.
*   **Easy Package Discovery:** Find packages through the web interface or search within the `nur-combined` repository.

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

1.  Include NUR in your `packageOverrides`:

    To make NUR accessible for your login user, add the following to `~/.config/nixpkgs/config.nix`:

    ```nix
    {
      packageOverrides = pkgs: {
        nur = import (builtins.fetchTarball "https://github.com/nix-community/NUR/archive/main.tar.gz") {
          inherit pkgs;
        };
      };
    }
    ```

    For NixOS add the following to your `/etc/nixos/configuration.nix`

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

Pinning NUR versions can prevent build failures.

```nix
builtins.fetchTarball {
  url = "https://github.com/nix-community/NUR/archive/3a6a6f4da737da41e27922ce2cfacf68a109ebce.tar.gz";
  sha256 = "04387gzgl8y555b3lkz9aiw9xsldfg4zmzp930m62qw8zbrvrshd";
}
```

## How to Use

Install packages from NUR using:

```bash
$ nix-shell -p nur.repos.mic92.hello-nur
```

```bash
$ nix-env -f '<nixpkgs>' -iA nur.repos.mic92.hello-nur
```

```nix
environment.systemPackages = with pkgs; [
  nur.repos.mic92.hello-nur
];
```

*Remember to verify packages before installation, as NUR packages are not reviewed by Nixpkgs maintainers.*

### Using a Single Package in a Devshell

Here's a simple example to add a single package from NUR to a devshell defined in a flake.nix.

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

### Using the Flake in NixOS

Using overlays and modules from NUR in your configuration is fairly straightforward.

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

Integrate with [Home Manager](https://github.com/rycee/home-manager) by adding your modules to the `imports` attribute.

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

## Adding Your Own Repository

1.  Create a repository with a `default.nix`. Use the [repository template](https://github.com/nix-community/nur-packages-template) as a starting point.
2.  Do NOT import packages using `with import <nixpkgs> {};`. Instead, use the `pkgs` argument.
3.  Your repository should return a set of Nix derivations.
4.  Use `nix-shell` or `nix-build` to test your packages.
5.  Add your repository to `repos.json` in the NUR repository.
6.  Run `./bin/nur format-manifest` and commit the changes to `repos.json` (not `repos.json.lock`).
7.  Create a pull request on the NUR GitHub repository.

###  Using a Different Root Expression

Use the `file` option to specify a file other than `default.nix`.

### Update NUR's Lock File

Use the service https://nur-update.nix-community.org/update?repo=<your-repo-name> after pushing updates.

### Why Are My NUR Packages Not Updating?

Ensure your repository's evaluation passes. Common issues include incorrect license attributes or using `builtins.fetchGit` instead of `pkgs.fetchgit`. Check the [latest build job](https://github.com/nix-community/NUR/actions) for details.

#### Local Evaluation Check

Run this command in your `nur-packages/` folder

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

### Git Submodules

Enable submodules in your repo definition:

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

*   **NixOS Modules:** Place modules in a `modules` attribute.
*   **Overlays:** Place overlays in an `overlays` attribute.
*   **Library Functions:** Put reusable functions in a `lib` attribute.

## Overriding Repositories

Override repositories with the `repoOverrides` argument.

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

### Overriding Repositories with Flake

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

Or with overlay

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

*   Ensure packages build and set `meta.broken = true` if they don't.
*   Provide comprehensive `meta` attributes, following the [Nixpkgs manual](https://nixos.org/nixpkgs/manual/#sec-standard-meta-attributes).
*   Keep repositories slim.
*   Reuse packages from Nixpkgs when possible.

**Examples for Packages that Could Be in NUR:**

*   Packages for a small audience
*   Pre-releases
*   Older versions of packages
*   Automatically generated packages
*   Software with specific patches
*   Experiments

## Contact

Join us on [#nur:nixos.org](https://matrix.to/#/#nur:nixos.org) or [Discourse](https://discourse.nixos.org/).