docker build -t caml -f Dockerfile .

docker run --rm -it \
    -v $(pwd):/caml \
    -v /caml/.venv \
    -v ~/.ssh:/root/.ssh \
    -w /caml \
    caml