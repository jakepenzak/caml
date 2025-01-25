# See here for image contents: https://github.com/microsoft/vscode-dev-containers/tree/v0.224.2/containers/python-3/.devcontainer/base.Dockerfile

ARG VARIANT="3.10-bullseye"
FROM mcr.microsoft.com/devcontainers/python:${VARIANT}

ARG UV_VERSION=0.5.23

# Install some base dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    fonts-powerline \
    ca-certificates \
    ssh \
    openjdk-17-jdk \
    && rm -rf /var/lib/apt/lists/*


# Add openjdk to path & set env variable
ENV JAVA_HOME="/usr/lib/jvm/java-17-openjdk-amd64"
ENV PATH="$JAVA_HOME/bin:$PATH"

# Change ownership to enable volume creation via devcontainer.json
USER root
RUN mkdir -p /home/vscode/.cache/uv && chown -R vscode:vscode /home/vscode/.cache

USER vscode

# Install uv
RUN pipx install uv==${UV_VERSION} && pipx ensurepath
ENV UV_HTTP_TIMEOUT=300

# Install startship
RUN curl -sS https://starship.rs/install.sh | sh -s -- -y
RUN echo 'eval "$(starship init bash)"' >> ~/.bashrc

# Ensure venv activation on opening terminal
RUN echo 'source .venv/bin/activate' >> ~/.bashrc
