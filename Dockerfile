FROM python:3.10-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    fonts-powerline \
    ca-certificates \
    nano \
    ssh \
    && rm -rf /var/lib/apt/lists/*

# Install uv
ADD https://astral.sh/uv/0.4.0/install.sh /uv-installer.sh
ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh
ENV PATH="/root/.cargo/bin/:$PATH"

# Install the project with intermediate layers
ADD .dockerignore .

# First, install the dependencies
WORKDIR /caml
ADD uv.lock /caml/uv.lock
ADD pyproject.toml /caml/pyproject.toml
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project
RUN  curl -sS https://starship.rs/install.sh | sh -s -- -y
RUN echo 'eval "$(starship init bash)"' >> ~/.bashrc

# Then, install the rest of the project
ADD . /caml
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen

# Place executables in the environment at the front of the path
ENV PATH="/caml/.venv/bin:$PATH"

RUN pre-commit install 

# Default command to open a bash shell
CMD ["/bin/bash"]