# NUR: Community-Driven Nix Package Repository

**Expand your Nix ecosystem with the Nix User Repository (NUR), a decentralized platform for sharing and discovering community-contributed packages.**  [Explore the original repository](https://github.com/nix-community/NUR).

**Key Features:**

*   **Community-Driven:** Access a vast library of packages from users like you.
*   **Decentralized:** Faster access to new packages and updates, bypassing the Nixpkgs review process.
*   **Automated Evaluation:** NUR automatically checks repositories for errors before integrating updates.
*   **Easy Installation:** Seamless integration with flakes, `packageOverrides`, NixOS, and Home Manager.
*   **Flexible Package Discovery:** Find packages via a web search, or the `nur-combined` repository.
*   **Modular Support:** Host NixOS modules, overlays, and library functions within your repositories.

## Installation

NUR offers multiple installation methods:

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

Use the overlay (`overlays.default`) or `legacyPackages.<system>`.

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

For NixOS, add this to your `/etc/nixos/configuration.nix`:

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

Pin the version to avoid caching issues and improve build reliability:

```nix
builtins.fetchTarball {
  url = "https://github.com/nix-community/NUR/archive/3a6a6f4da737da41e27922ce2cfacf68a109ebce.tar.gz";
  sha256 = "04387gzgl8y555b3lkz9aiw9xsldfg4zmzp930m62qw8zbrvrshd";
}
```

## How to Use

Once installed, use packages from the NUR namespace:

```bash
$ nix-shell -p nur.repos.mic92.hello-nur
nix-shell> hello
Hello, NUR!
```

Or:

```bash
$ nix-env -f '<nixpkgs>' -iA nur.repos.mic92.hello-nur
```

Or in `configuration.nix`:

```nix
environment.systemPackages = with pkgs; [
  nur.repos.mic92.hello-nur
];
```

**Important: Packages in NUR are not reviewed by Nixpkgs members; check expressions before installing.**

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

### Using NUR with NixOS

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

*   **Packages search for NUR:** [https://nur.nix-community.org/](https://nur.nix-community.org/)
*   **nur-combined:** [https://github.com/nix-community/nur-combined/search](https://github.com/nix-community/nur-combined/search)

## Adding Your Own Repository

1.  **Create a Repository:**  Create a Git repository with a `default.nix` file in its root, or use the [repository template](https://github.com/nix-community/nur-packages-template).
2.  **Nix Expression:**  Ensure each repository returns a set of Nix derivations:

    ```nix
    { pkgs }:
    {
      hello-nur = pkgs.callPackage ./hello-nur {};
    }
    ```

3.  **Example `default.nix` for a package:**

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
      doCheck = false;
      meta = with lib; {
        description = "A program that produces a familiar, friendly greeting";
        longDescription = ''...;
        homepage = https://www.gnu.org/software/hello/manual/;
        changelog = "https://git.savannah.gnu.org/cgit/hello.git/plain/NEWS?h=v${version}";
        license = licenses.gpl3Plus;
        maintainers = [ maintainers.eelco ];
        platforms = platforms.all;
      };
    }
    ```

4.  **Build and Test:** Use `nix-shell` or `nix-build` to test:

    ```bash
    $ nix-shell --arg pkgs 'import <nixpkgs> {}' -A hello-nur
    nix-shell> hello
    Hello, NUR!
    ```

5.  **Set `pkgs` argument:**  For development convenience, set a default value for the `pkgs` argument:

    ```nix
    { pkgs ? import <nixpkgs> {} }:
    {
      hello-nur = pkgs.callPackage ./hello-nur {};
    }
    ```
6.  **Add Your Repository to NUR:**
    *   Clone the NUR repository:

        ```bash
        $ git clone --depth 1 https://github.com/nix-community/NUR
        $ cd NUR
        ```
    *   Edit `repos.json`:

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
    *   Run `./bin/nur format-manifest` to sort.
    *   Add and commit the changes to `repos.json` (but NOT `repos.json.lock`).
    *   Open a pull request to [https://github.com/nix-community/NUR](https://github.com/nix-community/NUR).

    *   **Important:** URLs must point to a Git repository.
7.  **Using `file` for your root nix file:**
    *   Set the `file` option to a path relative to the repository root:

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

### Updating NUR's lock file after updating your repository

To update NUR faster, use our service at https://nur-update.nix-community.org/
after you have pushed an update to your repository, e.g.:

```console
curl -XPOST https://nur-update.nix-community.org/update?repo=mic92
```

Check out the [github page](https://github.com/nix-community/nur-update#nur-update-endpoint) for further details

### Why are my NUR packages not updating?

Repository evaluation errors will prevent updates. Common issues:

*   Incorrect license attributes in metadata.
*   Using built-in fetchers during evaluation (use `pkgs.fetch*` instead).

Check the [latest build job](https://github.com/nix-community/NUR/actions) to see if your repository builds correctly.

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

### NixOS modules, overlays and library function support

Place NixOS modules in the `modules` attribute:

```nix
{ pkgs }: {
  modules = import ./modules;
}
```

Modules should be defined as paths, not functions, to avoid conflicts if imported from multiple locations.
A module with no [_class](https://nixos.org/manual/nixpkgs/stable/index.html#module-system-lib-evalModules-param-class) will be assumed to be both a NixOS and Home Manager module.
If a module is NixOS or Home Manager specific, the `_class` attribute should be set to `"nixos"` or [`"home-manager"`](https://github.com/nix-community/home-manager/commit/26e72d85e6fbda36bf2266f1447215501ec376fd).

For overlays, use the `overlays` attribute:

```nix
# default.nix
{
  overlays = {
    hello-overlay = import ./hello-overlay;
  };
}
```

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

Override repositories using the `repoOverrides` argument:

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

*   **Build and Meta:** Ensure packages build and set `meta.broken = true` if not. Include [Nixpkgs manual](https://nixos.org/nixpkgs/manual/#sec-standard-meta-attributes) meta attributes.
*   **Keep it Slim:** Minimize repository size and download times.
*   **Reuse:**  Leverage Nixpkgs packages where possible.

**Ideal Use Cases:**

*   Packages for small audiences.
*   Pre-releases.
*   Older versions of packages.
*   Automated package sets.
*   Software with opinionated patches.
*   Experiments.

## Contact

*   Matrix: [#nur:nixos.org](https://matrix.to/#/#nur:nixos.org)
*   Discourse: [https://discourse.nixos.org/](https://discourse.nixos.org/)