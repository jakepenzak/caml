docker build -t caml -f Dockerfile .

docker run --rm -it \
    -v $(pwd):/caml \
    -v /caml/.venv \
    -v ~/.ssh:/root/.ssh \
    -w /caml \
    caml bash -c "chown -R root:root /root/.ssh && chmod 700 /root/.ssh && chmod 600 /root/.ssh/id_rsa && chmod 600 /root/.ssh/config && bash"
