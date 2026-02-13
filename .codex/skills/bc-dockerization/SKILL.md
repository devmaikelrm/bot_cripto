---
name: bc-dockerization
description: Build and maintain Bot Cripto Docker images for training and inference with reproducible runtime behavior. Use for containerization and image release tasks.
---

# bc-dockerization

1. Maintain `docker/Dockerfile.train` and `docker/Dockerfile.infer`.
2. Keep image entrypoints aligned with CLI commands.
3. Ensure dependencies are installed from project metadata.
4. Verify containers run with mounted data/models/logs directories.
5. Keep `scripts/build_and_push.sh` updated with image names/tags.
