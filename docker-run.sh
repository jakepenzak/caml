# Accept user email and username as options to pass into git config

# Usage: ./docker-run.sh -e <email> -u <username>

email=""
username=""
while getopts e:u: flag
do
    case "${flag}" in
        e) email=${OPTARG};;
        u) username=${OPTARG};;
        *) echo "Usage: $0 -e <email> -u <username>"; exit 1;;
    esac
done

# Check if both email and username are provided
if [ -z "$email" ] || [ -z "$username" ]; then
    echo "Both email and username must be provided."
    echo "Usage: $0 -e <email> -u <username>"
    exit 1
fi

docker build -t caml -f Dockerfile .

docker run --rm -it \
    -v $(pwd):/caml \
    -v /caml/.venv \
    -w /caml \
    caml \
    bash -c "git config --global --add safe.directory /caml && git config --global user.email $email && git config --global user.name $username && bash"
