# AcousticPose Anonymous Project Page

This repository is a static GitHub Pages site for the anonymous AcousticPose project page.

The page contains:

- synchronized held-out visual proof reels
- paper-facing result figures
- code/data reproducibility section
- downloadable code script archive
- public-dataset provenance for MELD, CREMA-D, and RAVDESS

## Deploy on GitHub Pages

1. Create a new GitHub repository.
2. Upload every file in this folder to the repository root.
3. In the repository settings, open `Pages`.
4. Set `Build and deployment` to `GitHub Actions`.
5. Push to `main`.

The included workflow at `.github/workflows/pages.yml` deploys the static site automatically.

After the first deployment, the permanent URL will be:

```text
https://<github-username>.github.io/<repository-name>/
```

Use that URL in the manuscript instead of any `trycloudflare.com` link.

## Local Preview

From this folder:

```bash
python3 -m http.server 4173
```

Then open:

```text
http://localhost:4173
```

## Notes

The videos are held-out demonstration assets generated from public datasets. Raw dataset videos are not redistributed here; use the original MELD, CREMA-D, and RAVDESS providers for raw data access.
