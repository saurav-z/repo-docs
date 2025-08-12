# Nix User Repository (NUR): Expand Your Nix Ecosystem

**Enhance your Nix experience with the Nix User Repository (NUR), a community-driven collection of user-contributed Nix packages, modules, and overlays.**

[View the original repository](https://github.com/nix-community/NUR)

NUR provides a decentralized platform for Nix enthusiasts to share and discover packages, offering access to software not yet available in the official Nixpkgs. It's a dynamic space for exploring cutting-edge software, pre-releases, and custom configurations.

**Key Features:**

*   **Community-Driven:** Benefit from a vast collection of packages and configurations contributed by Nix users.
*   **Decentralized:** Easily access and install packages from user-maintained repositories.
*   **Flexible Installation:** Integrates seamlessly with flakes, `packageOverrides`, and Home Manager.
*   **Automatic Checks:** Ensures repository updates and evaluation checks before propagation.
*   **Customization:** Easily add your own repository and contribute to the community.
*   **Modules, Overlays, and Library Functions:** Share and discover NixOS modules, overlays, and reusable library functions.

## Installation

NUR can be integrated into your Nix configuration in a few ways:

### Using Flakes

Include NUR in your `flake.nix` to leverage the latest features.

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

### Using `packageOverrides`

Add NUR to your `packageOverrides` to make packages available for your user.
See original README for exact code, as it varies based on setup.

### Pinning

Pin specific versions of NUR to ensure consistent builds.

```nix
builtins.fetchTarball {
  # Get the revision by choosing a version from https://github.com/nix-community/NUR/commits/main
  url = "https://github.com/nix-community/NUR/archive/3a6a6f4da737da41e27922ce2cfacf68a109ebce.tar.gz";
  # Get the hash by running `nix-prefetch-url --unpack <url>` on the above url
  sha256 = "04387gzgl8y555b3lkz9aiw9xsldfg4zmzp930m62qw8zbrvrshd";
}
```

## How to Use

Install or reference packages from the NUR namespace.

```console
$ nix-shell -p nur.repos.mic92.hello-nur
```

or

```console
$ nix-env -f '<nixpkgs>' -iA nur.repos.mic92.hello-nur
```

or

```nix
environment.systemPackages = with pkgs; [
  nur.repos.mic92.hello-nur
];
```

**Important Note:** Packages in NUR are not reviewed by Nixpkgs maintainers. Exercise caution and review packages before installation.

## Finding Packages

*   **Packages search for NUR:** Explore the vast collection of NUR packages.
*   **nur-combined repository:** Search the [nur-combined](https://github.com/nix-community/nur-combined) repository, which contains all nix expressions from all users, via [github](https://github.com/nix-community/nur-combined/search).

## Contributing

### How to add your own repository.

1.  Create a repository with a `default.nix` file.
2.  Define packages within the `default.nix` using a structure like:

    ```nix
    { pkgs }:
    {
      hello-nur = pkgs.callPackage ./hello-nur {};
    }
    ```

3.  Add your repository details to the `repos.json` file in the NUR repository:

    ```json
    {
        "repos": {
            "mic92": {
                "url": "https://github.com/Mic92/nur-packages"
            },
            "<your-repo-name>": {
                "url": "https://github.com/<your-user>/<your-repo>"
            }
        }
    }
    ```

4.  Submit a pull request to the NUR repository.

### Using a different nix file as root expression

To use a different file instead of `default.nix` to load packages from, set the `file` option to a path relative to the repository root:

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

To update NUR faster, you can use our service at https://nur-update.nix-community.org/
after you have pushed an update to your repository, e.g.:

```console
curl -XPOST https://nur-update.nix-community.org/update?repo=mic92
```

Check out the [github page](https://github.com/nix-community/nur-update#nur-update-endpoint) for further details

### Why are my NUR packages not updating?

*   Ensure your evaluation succeeds by checking the [latest build job](https://github.com/nix-community/NUR/actions).
*   Resolve common errors like incorrect license attributes or using `builtins.fetch*` functions instead of `pkgs.fetch*`.

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

*   Place NixOS modules in the `modules` attribute.
*   Define overlays using the `overlays` attribute.
*   Share library functions in the `lib` attribute.

## Overriding Repositories

You can override repositories with the `repoOverrides` argument.

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

*   Ensure packages build and set the `meta.broken` attribute to `true` if not working.
*   Include appropriate `meta` attributes, following the [Nixpkgs manual](https://nixos.org/nixpkgs/manual/#sec-standard-meta-attributes).
*   Keep repositories lean and efficient.
*   Prioritize reusing packages from Nixpkgs.

## Contact

For support and discussions, join our [Matrix channel](https://matrix.to/#/#nur:nixos.org) or participate on [Discourse](https://discourse.nixos.org/).