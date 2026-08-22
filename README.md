# Immutable Leakage-Safe Corpus Service

Vercel-ready implementation of `POST /build-corpus`.

## Deploy

1. Create a new GitHub repository and upload the contents of this folder (not the outer folder).
2. In Vercel, choose **Add New > Project**, import the repository, and click **Deploy**.
3. Leave Framework Preset as **Other** and do not add environment variables.
4. Submit the Vercel production URL, for example `https://your-project.vercel.app`.

The grader will call `POST https://your-project.vercel.app/build-corpus`.

## Local tests

```bash
python -m unittest discover -s tests -v
```
