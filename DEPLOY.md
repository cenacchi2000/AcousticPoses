# Permanent Hosting Instructions

The old `trycloudflare.com` URL was a temporary tunnel and should not be used in a manuscript.

For a permanent anonymous website:

1. Create a public GitHub repository, for example `acousticpose-demo`.
2. Add all files from this folder to the repository root.
3. Commit and push to `main`.
4. In GitHub, go to `Settings -> Pages`.
5. Select `GitHub Actions` as the source.
6. Wait for the `Deploy static GitHub Pages site` action to complete.

The permanent URL will have this form:

```text
https://<github-username>.github.io/acousticpose-demo/
```

If the repository is named `<github-username>.github.io`, the permanent URL becomes:

```text
https://<github-username>.github.io/
```

For anonymous review, use a neutral GitHub account and a non-identifying repository name.
