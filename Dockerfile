FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    fonts-powerline \
    ca-certificates \
    nano \
    ssh \
    openjdk-17-jdk \
    && rm -rf /var/lib/apt/lists/*

# Add openjdk to path & set env variable
ENV JAVA_HOME="/usr/lib/jvm/java-17-openjdk-amd64"
ENV PATH="$JAVA_HOME/bin:$PATH"

# Install uv
ADD https://astral.sh/uv/0.4.10/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh
ENV PATH="/root/.cargo/bin/:$PATH"

# Install the project with intermediate layers
ADD .dockerignore .

# First, install the dependencies
WORKDIR /caml
ADD uv.lock /caml/uv.lock
ADD pyproject.toml /caml/pyproject.toml
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --all-extras --no-install-project
RUN  curl -sS https://starship.rs/install.sh | sh -s -- -y
RUN echo 'eval "$(starship init bash)"' >> ~/.bashrc

# Then, install the rest of the project
ADD . /caml
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --all-extras --frozen

# Expose port for quarto
EXPOSE 8000

# Copy keys for github ssh access
COPY --chown=root:root .github_access /root/.ssh

# Place executables in the environment at the front of the path
ENV PATH="/caml/.venv/bin:$PATH"
