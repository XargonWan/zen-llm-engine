ARG TARGETPLATFORM
FROM ghcr.io/astral-sh/uv:latest AS uv_source

FROM ghcr.io/linuxserver/baseimage-selkies:ubuntunoble

# Provided automatically by Docker Buildx (e.g. "amd64", "arm64"). Re-declared
# here so it is available inside this build stage for arch-aware COPY below.
ARG TARGETARCH

# --- Webtop / Zen environment setup ---
ENV TITLE="Zen LLM Engine"
ENV PIXELFLUX_USE_XSHM=0 \
    PIXELFLUX_DISABLE_XSHM=1 \
    PIXELFLUX_NO_XSHM=1 \
    QT_X11_NO_MITSHM=1 \
    DISABLE_XSHM=1 \
    BROWSER=/usr/local/bin/chromium-browser
ENV REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt

# Inject uv binary from astral image
COPY --from=uv_source /uv /usr/local/bin/uv

COPY webtop/s6-services/uvicorn /etc/services.d/uvicorn
# Install system packages and browser deps
RUN echo 'Package: snapd' > /etc/apt/preferences.d/no-snap && \
    echo 'Pin: release a=*' >> /etc/apt/preferences.d/no-snap && \
    echo 'Pin-Priority: -10' >> /etc/apt/preferences.d/no-snap && \
    apt-get update && apt-get purge -y snapd && \
    apt-get autoremove -y && rm -rf /snap /var/snap /var/lib/snapd && \
        apt-get install -y --no-install-recommends \
            xz-utils \
      python3 python3-venv python3-pip \
      git curl wget unzip nano vim \
      lsb-release ca-certificates openssl \
      htop net-tools iputils-ping \
      ffmpeg mariadb-client libmariadb3 libmariadb-dev \
      espeak-ng libespeak-ng1 \
      xorg dbus-x11 x11-xserver-utils \
      xfce4 xfce4-goodies xfce4-terminal thunar mousepad ristretto \
      adwaita-icon-theme util-linux dbus-x11 at-spi2-core \
      pulseaudio pulseaudio-utils pavucontrol \
      fonts-liberation libnss3 libxss1 libappindicator3-1 libatk-bridge2.0-0 \
      libgtk-3-0 libgbm-dev libasound2t64 xvfb x11vnc fluxbox novnc python3-websockify \
      ca-certificates && \
    update-ca-certificates --fresh && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install gemini-cli (optional helper, same as Synthetic Heart)
RUN pip3 install --no-cache-dir gemini-cli || true

# Install Chromium 148 fully offline from vendored .deb files.
#
# zendriver talks to a real, unmodified Chromium directly over the Chrome
# DevTools Protocol (CDP) — there is no separate chromedriver binary at all,
# so chromium-driver is no longer installed here (previously required to
# keep undetected-chromedriver's uc.Chrome stealth patching working, and
# version-pinned in lockstep with it). The 148 pin on chromium/chromium-common
# itself is kept unchanged for this pass — it was not re-evaluated as part of
# the zendriver migration and bumping it is a separate decision. Debian
# bookworm-security no longer ships 148 (only 150 is available now), so we
# vendor the exact 148 .deb packages in the repo and install them from disk.
#
# The .deb packages are Debian bookworm builds, but the base image is Ubuntu
# Noble, so their runtime deps use Debian package names that Noble lacks
# (libdav1d6, libdouble-conversion3, libharfbuzz-subset0, libjpeg62-turbo,
# libminizip1, libopenh264-7, libxnvctrl0). Rather than depend on the Debian
# repos being reachable at build time, we ALSO vendor those dependency .deb
# files under vendor/chromium148/<arch>/deps/ so the entire install is offline
# and reproducible even if the Debian bookworm repos ever disappear.
#
# The vendored packages are split per architecture (amd64/, arm64/) because the
# multi-arch build produces both linux/amd64 and linux/arm64 images. TARGETARCH
# selects the matching set at build time.
#
# Pinned: chromium / chromium-common 148.0.7778.215-1~deb12u1
COPY vendor/chromium148/${TARGETARCH}/ /tmp/chromium148/
RUN apt-get update && \
    apt-get purge -y google-chrome google-chrome-stable || true && \
    # Install the vendored Debian-flavoured runtime deps first (offline), then
    # the Chromium 148 packages. dpkg resolves the local files with no
    # network access. The trailing apt-get -f is a safety net that only pulls
    # from whatever repos the base image already trusts (no Debian repos added).
    dpkg -i /tmp/chromium148/deps/*.deb || true && \
    dpkg -i /tmp/chromium148/chromium-common.deb \
            /tmp/chromium148/chromium.deb || \
    apt-get install -y -f --no-install-recommends && \
    apt-mark hold chromium chromium-common && \
    rm -rf /tmp/chromium148 && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    # Fail the build loudly if the browser did not land (previously masked by `|| true`).
    chromium --version


# Browser-level account sign-in off (BrowserSignin=0, SyncDisabled). A website
# login (Gemini, ...) must stay plain cookies: with Chromium's own sign-in on,
# its account reconciler ran at every launch and the web login was gone right
# after a relaunch. See ZendriverLLMBase._ensure_profile_preferences.
COPY webtop/chromium-policies/zen.json /etc/chromium/policies/managed/zen.json

# Chromium profile setup (in /config/.config/chromium-synth — matches SyntH)
RUN mkdir -p '/config/.config/chromium-synth' && \
    chown -R abc:abc /config && \
    chmod -R 775 /config && \
    mkdir -p /usr/local/share/applications

RUN cat > /usr/local/share/applications/chromium-synth.desktop <<'EOF'
[Desktop Entry]
Version=1.0
Name=Chromium SyntH
Exec=/usr/bin/chromium --no-sandbox --user-data-dir=/config/.config/chromium-synth %U
Terminal=false
Type=Application
Categories=Network;WebBrowser;
EOF

RUN mkdir -p /config/.local/share/applications && \
    cp /usr/local/share/applications/chromium-synth.desktop /config/.local/share/applications/ && \
    chown -R abc:abc /config/.local

WORKDIR /app
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

ENV ZEN_LLM_DB=/app/data/zen_engine.db
ENV CHROMIUM_HEADLESS=0
ENV PYTHONUNBUFFERED=1

RUN echo xfce4-session > /config/desktop-session
# S6 Websockify
COPY webtop/s6-services/websockify /etc/s6-overlay/s6-rc.d/websockify
RUN chmod +x /etc/s6-overlay/s6-rc.d/websockify/run && \
    echo 'longrun' > /etc/s6-overlay/s6-rc.d/websockify/type && \
    mkdir -p /etc/s6-overlay/s6-rc.d/user/contents.d && \
    echo websockify > /etc/s6-overlay/s6-rc.d/user/contents.d/websockify && \
    chown -R abc:abc /etc/s6-overlay/s6-rc.d/websockify

# Final cleanup
RUN mv /usr/bin/thunar /usr/bin/thunar-real && \
  rm -f /etc/xdg/autostart/xfce4-power-manager.desktop /etc/xdg/autostart/xscreensaver.desktop && \
  rm -rf /tmp/*

