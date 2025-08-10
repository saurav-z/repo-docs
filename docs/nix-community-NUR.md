# NUR: Your Community Hub for Nix Packages

**NUR (Nix User Repository) is a community-driven repository that allows you to quickly access and install user-contributed Nix packages.** Check out the original repository [here](https://github.com/nix-community/NUR).

## Key Features

*   **Community-Driven:** Access packages created and maintained by the Nix community.
*   **Decentralized:** Get access to new packages faster than through Nixpkgs.
*   **Flexible Installation:** Integrate with flakes, `packageOverrides`, NixOS modules, Home Manager, and devshells.
*   **Automated Checks:** NUR performs evaluation checks to ensure repository updates are valid.
*   **Package Discovery:** Easily find packages using the [Packages search for NUR](https://nur.nix-community.org/) or the [nur-combined](https://github.com/nix-community/nur-combined) repository.
*   **NixOS Module, Overlay and Library Function Support**: Provide NixOS modules, overlays and library functions within your repository.

## Installation

### Using Flakes

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

For NixOS, add the following to your `/etc/nixos/configuration.nix`

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

Pin the version using `builtins.fetchTarball` with `sha256` for caching:

```nix
builtins.fetchTarball {
  url = "https://github.com/nix-community/NUR/archive/3a6a6f4da737da41e27922ce2cfacf68a109ebce.tar.gz";
  sha256 = "04387gzgl8y555b3lkz9aiw9xsldfg4zmzp930m62qw8zbrvrshd";
}
```

## How to Use

Install and use packages from the NUR namespace:

```bash
$ nix-shell -p nur.repos.mic92.hello-nur
$ nix-env -f '<nixpkgs>' -iA nur.repos.mic92.hello-nur
# configuration.nix
environment.systemPackages = with pkgs; [
  nur.repos.mic92.hello-nur
];
```

***Important Note:*** *Always check expressions before installing, as NUR does not regularly check for malicious content.*

## Example Usage: Single Package in Devshell

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

## Integration with NixOS

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

Add modules to the `imports` attribute:

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

1.  Create a repository with a `default.nix` file. Use the [repository template](https://github.com/nix-community/nur-packages-template) for structure.
2.  Reference dependencies from `pkgs`, not `<nixpkgs>`.
3.  Return a set of Nix derivations.
4.  Add your repository to `repos.json` in the NUR repository.
5.  Run `./bin/nur format-manifest` to format the JSON file and ensure it is sorted alphabetically
6.  Commit the changes and open a pull request.

### Using a different nix file as root expression

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

### Updating the NUR Lock File

Use the NUR update service after updating your repository:

```bash
curl -XPOST https://nur-update.nix-community.org/update?repo=mic92
```

### Troubleshooting Package Updates

*   Ensure evaluations in the [latest build job](https://github.com/nix-community/NUR/actions) succeed.
*   Common errors: incorrect metadata, or usage of `fetch` without a `pkgs.` prefix.

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

## NixOS Modules, Overlays, and Library Functions

*   **Modules:** Place NixOS modules in the `modules` attribute within your `default.nix`.

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
*   **Overlays:** Define overlays in the `overlays` attribute.

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
*   **Library Functions:** Put reusable nix functions in the `lib` attribute.

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

Use `repoOverrides` to test changes before publishing:

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

*   Ensure packages build and set `meta.broken = true` if broken.
*   Use standard [Nixpkgs meta attributes](https://nixos.org/nixpkgs/manual/#sec-standard-meta-attributes).
*   Keep repositories concise.
*   Leverage Nixpkgs for dependencies.

### Examples of suitable packages:

*   Packages for small audiences
*   Pre-releases
*   Old package versions
*   Automatically generated package sets
*   Software with custom patches
*   Experiments

## Contact

*   Matrix: [#nur:nixos.org](https://matrix.to/#/#nur:nixos.org)
*   Discourse: [https://discourse.nixos.org](https://discourse.nixos.org)