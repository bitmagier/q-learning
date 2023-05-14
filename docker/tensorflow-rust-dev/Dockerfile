FROM tensorflow/tensorflow:2.12.0-gpu
MAINTAINER bitmagier@mailbox.org

ARG DEV_USER
ARG SSH_AUTHORIZED_KEY

RUN echo "y\nY" | unminimize

ENV DEV_USER=bitmagier

RUN apt-get update
# RUN apt-get upgrade -y

RUN useradd --uid 1000 --create-home --shell /bin/bash $DEV_USER
RUN apt-get install -y sudo
RUN echo >> /etc/sudoers
RUN echo "$DEV_USER ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

RUN apt-get install -y openssh-server && \
echo "PasswordAuthentication no" >> /etc/ssh/sshd_config

USER 1000
RUN mkdir -p /home/$DEV_USER/.ssh && \
echo "$SSH_AUTHORIZED_KEY" > /home/$DEV_USER/.ssh/authorized_keys && \
chmod -R 700 /home/$DEV_USER/.ssh

USER 0
RUN apt-get install -y wget
ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:$PATH \
    RUST_VERSION=1.69.0

# from Dockerfile of rust-lang/docker-rust: https://github.com/rust-lang/docker-rust
RUN set -eux; \
    dpkgArch="$(dpkg --print-architecture)"; \
    case "${dpkgArch##*-}" in \
        amd64) rustArch='x86_64-unknown-linux-gnu'; rustupSha256='bb31eaf643926b2ee9f4d8d6fc0e2835e03c0a60f34d324048aa194f0b29a71c' ;; \
        armhf) rustArch='armv7-unknown-linux-gnueabihf'; rustupSha256='6626b90205d7fe7058754c8e993b7efd91dedc6833a11a225b296b7c2941194f' ;; \
        arm64) rustArch='aarch64-unknown-linux-gnu'; rustupSha256='4ccaa7de6b8be1569f6b764acc28e84f5eca342f5162cd5c810891bff7ed7f74' ;; \
        i386) rustArch='i686-unknown-linux-gnu'; rustupSha256='34392b53a25c56435b411d3e575b63aab962034dd1409ba405e708610c829607' ;; \
        *) echo >&2 "unsupported architecture: ${dpkgArch}"; exit 1 ;; \
    esac; \
    url="https://static.rust-lang.org/rustup/archive/1.25.2/${rustArch}/rustup-init"; \
    wget "$url"; \
    echo "${rustupSha256} *rustup-init" | sha256sum -c -; \
    chmod +x rustup-init; \
    ./rustup-init -y --no-modify-path --profile minimal --default-toolchain $RUST_VERSION --default-host ${rustArch}; \
    rm rustup-init; \
    chmod -R a+w $RUSTUP_HOME $CARGO_HOME; \
    rustup --version; \
    cargo --version; \
    rustc --version;

RUN echo "source /usr/local/cargo/env" >> /home/$DEV_USER/.profile
RUN su - bitmagier -c 'rustup default stable'

USER 0
RUN apt-get install -y libssl-dev less vim libnvidia-compute-525-server

EXPOSE 22/tcp

CMD /etc/init.d/ssh start && \
sleep infinity
