# FROM — base image. You start with someone else's work.
# python:3.12-slim = Python 3.12 on Linux, stripped of extras.
# WHY slim? Full python:3.12 = 1GB. slim = 130MB. Faster to build, push, pull.

FROM python:3.12-slim

# ─────────────────────────────────────────────
# RUN — execute a shell command DURING BUILD (not at runtime).
# We need libpq-dev because psycopg2 compiles C code that talks to PostgreSQL.
# gcc = C compiler needed for that compilation.
# rm -rf /var/lib/apt/lists/* = delete apt cache so it doesn't bloat the image.
# && chains commands into ONE RUN = ONE Docker layer = smaller image.

RUN apt-get update && apt-get install -y \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# ─────────────────────────────────────────────
# WORKDIR — all following commands run from this directory inside the container.
# Creates the directory if it doesn't exist.
# Think of it as: cd /app (but inside the container)

WORKDIR /app

# ─────────────────────────────────────────────
# COPY requirements FIRST — before copying your code.
# WHY? Docker builds in layers. Each instruction = one layer.
# If requirements.txt hasn't changed → Docker reuses the cached pip install layer.
# If you COPY . . first → any code change invalidates the pip install cache.
# This one ordering trick saves 2-3 minutes on every build.

COPY requirements.txt .

# ─────────────────────────────────────────────
# RUN pip install — installs all packages into the container.
# --no-cache-dir = don't store pip download cache in image (saves ~50MB).

RUN pip install --no-cache-dir -r requirements.txt

# ─────────────────────────────────────────────
# COPY app/ . — copy ONLY the app/ folder contents into /app in container.
# WHY not COPY . . ?
#   context is . (root), so COPY . . puts main.py at /app/app/main.py
#   then uvicorn main:app can't find it because it looks in /app directly.
# COPY app/ . copies the *contents* of app/ → main.py lands at /app/main.py ✓
# This also keeps frontend/, .env, notes.txt etc out of the FastAPI image.

COPY app/ .

# ─────────────────────────────────────────────
# EXPOSE — documents which port the app uses. Does NOT open the port.
# Actual port opening happens with -p flag in docker run or in docker-compose.

EXPOSE 8000

# ─────────────────────────────────────────────
# CMD — what runs when the container starts.
# Array syntax ["cmd", "arg"] = exec form. Signals go directly to the process.
# String syntax "cmd arg" = shell form. Wrapped in /bin/sh -c. Signals get lost.
# ALWAYS use array syntax for CMD so Ctrl+C and k8s SIGTERM work correctly.

CMD ["uvicorn","main:app","--host","0.0.0.0","--port","8000"]

# --host 0.0.0.0 = listen on ALL network interfaces inside container.
# Without this: uvicorn listens on 127.0.0.1 = only reachable from inside the container.
# With this: Docker can route external traffic to uvicorn.

