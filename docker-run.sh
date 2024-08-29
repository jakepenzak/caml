if [ -s 1 ]; then
    MOUNT_SSH="-v ~/.ssh:/root/.ssh"
else
    MOUNT_SSH=""
fi

docker build -t caml -f Dockerfile .

docker run --rm -it \
    -v $(pwd):/caml \
    -v /caml/.venv \
    $MOUNT_SSH \
    -w /caml \
    caml