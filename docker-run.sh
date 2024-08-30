#!/bin/bash

# Initialize variables for email, username, and use_git flag
email=""
username=""
use_git=0
image_name="caml"
dockerfile="Dockerfile"

# Function to display usage
usage() {
    echo "Usage: $0 [-g] -e <email> -u <username> [-i <image_name>] [-f <Dockerfile>]"
    echo "  -g, --use-git           Configure Git with the provided email and username"
    echo "  -e, --email             Email for Git configuration"
    echo "  -u, --username          Username for Git configuration"
    echo "  -i, --image-name        Name of the Docker image (default: my-docker-image)"
    echo "  -f, --dockerfile        Dockerfile to use for building the image (default: Dockerfile)"
    echo "  -h, --help              Display the help message"
    exit 1
}

# Parse command-line options
while [[ $# -gt 0 ]]; do
    case "$1" in
        -e|--email)
            email="$2"
            shift 2
            ;;
        -u|--username)
            username="$2"
            shift 2
            ;;
        -g|--use-git)
            use_git=1
            shift 1
            ;;
        -i|--image-name)
            image_name="$2"
            shift 2
            ;;
        -f|--dockerfile)
            dockerfile="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Check if both email and username are provided when use_git is enabled
if [[ $use_git -eq 1 ]]; then
    if [[ -z "$email" || -z "$username" ]]; then
        echo "Error: Both email and username must be provided when using Git. Use the -h option for help."
        exit 1
    fi
fi

# Build the Docker image
echo "Building Docker image '$image_name' using Dockerfile '$dockerfile'..."
docker build -t "$image_name" -f "$dockerfile" .

# Check if the build was successful
if [[ $? -ne 0 ]]; then
    echo "Error: Docker build failed."
    exit 1
fi

# Build the Docker run command
docker_cmd="docker run -p 8000:8000 -it --rm -v $(pwd):/caml -v /caml/.venv -w /caml"

# Add the image name and bash command
docker_cmd+=" $image_name bash"

# Configure Git if use_git is enabled
if [[ $use_git -eq 1 ]]; then
    docker_cmd+=" -c 'git config --global --add safe.directory /caml && git config --global user.name '$username' && git config --global user.email '$email' && pre-commit install && bash'"
else
    docker_cmd+=" -c 'bash'"
fi

eval $docker_cmd
