# Nix User Repository (NUR): Unleash Community-Driven Nix Packages

**NUR empowers Nix users with a decentralized way to discover, share, and install packages directly from the community. [Explore the NUR project on GitHub](https://github.com/nix-community/NUR).**

## Key Features

*   **Community-Driven:** Access a vast collection of user-contributed Nix packages.
*   **Decentralized:** Discover and install packages faster than traditional Nixpkgs.
*   **Automated Updates:** NUR automatically checks and validates package repositories.
*   **Flexible Installation:** Integrate seamlessly with flakes, `packageOverrides`, NixOS configurations, and Home Manager.
*   **Package Search:** Easily find packages through the NUR package search or the combined repository.
*   **Repository Templates:** Simplify package creation with the provided template.
*   **Supports NixOS Modules, Overlays & Library Functions**:  Extend your NixOS configurations with modules, overlays and library functions.
*   **Repo Overrides:** Test your changes before publishing.

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

Then, use the overlay (`overlays.default`) or `legacyPackages.<system>`.

### Using `packageOverrides`

Add the following to `~/.config/nixpkgs/config.nix` or `/etc/nixos/configuration.nix`:

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

Pin the version for faster builds:

```nix
builtins.fetchTarball {
  url = "https://github.com/nix-community/NUR/archive/3a6a6f4da737da41e27922ce2cfacf68a109ebce.tar.gz";
  sha256 = "04387gzgl8y555b3lkz9aiw9xsldfg4zmzp930m62qw8zbrvrshd";
}
```

## How to Use

Install packages from the NUR namespace:

```bash
nix-shell -p nur.repos.mic92.hello-nur
```

or

```bash
nix-env -f '<nixpkgs>' -iA nur.repos.mic92.hello-nur
```

or

```nix
# configuration.nix
environment.systemPackages = with pkgs; [
  nur.repos.mic92.hello-nur
];
```

**Important Note:** Carefully review package expressions before installing, as packages are not reviewed by Nixpkgs maintainers.

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

Add your modules to the `imports` attribute in Home Manager:

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
*   [nur-combined](https://github.com/nix-community/nur-combined) repository

## Adding Your Own Repository

1.  Create a Git repository with a `default.nix`.
2.  Follow the provided [repository template](https://github.com/nix-community/nur-packages-template).
3.  Do not import packages using `<nixpkgs>`. Instead, use the provided `pkgs` argument.
4.  Ensure your repository returns a set of Nix derivations:

```nix
{ pkgs }:
{
  hello-nur = pkgs.callPackage ./hello-nur {};
}
```

5.  Add your repository details to `repos.json` in the NUR repository.
6.  Run `./bin/nur format-manifest` and commit the changes.
7.  Open a pull request to [https://github.com/nix-community/NUR](https://github.com/nix-community/NUR).

### Using a Different Root Expression

Specify a different file with the `file` option:

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

Update the NUR's lock file by pushing updates to your repository and using the NUR update service:

```bash
curl -XPOST https://nur-update.nix-community.org/update?repo=mic92
```

### Troubleshooting Package Updates

*   Ensure the evaluation process completes without errors. Check the [latest build job](https://github.com/nix-community/NUR/actions).
*   Common errors involve using incorrect license attributes or built-in fetchers.

#### Local Evaluation Check

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

### Git Submodules

Enable submodules with `submodules`:

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

*   **NixOS Modules:** Place in the `modules` attribute within your repository.
*   **Overlays:** Use the `overlays` attribute.
*   **Library Functions:** Put reusable functions in the `lib` attribute.

### Overriding Repositories

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

### Overriding Repositories with Flake

**Experimental**, override repositories in two ways:
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

*   Build packages and set `meta.broken = true` for broken packages.
*   Include standard `meta` attributes for package discovery.
*   Keep repositories lean and focused.
*   Reuse packages from Nixpkgs whenever possible.

**Ideal NUR Package Candidates:**

*   Packages for a small audience
*   Pre-release software
*   Older versions
*   Auto-generated package sets
*   Software with custom patches
*   Experimental software

## Contact

*   Matrix: [#nur:nixos.org](https://matrix.to/#/#nur:nixos.org)
*   Discourse: [https://discourse.nixos.org/](https://discourse.nixos.org/)