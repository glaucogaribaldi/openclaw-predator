#!/usr/bin/env bash
set -euo pipefail

echo "=================================================================="
echo "⚡ Installing TRE PREDATOR v1.0 (Predictive & Multi-Agent Quant OS)"
echo "=================================================================="

INSTALL_DIR="${HOME}/openclaw-predator"
REPO_URL="https://github.com/glaucogaribaldi/openclaw-predator.git"

if [ ! -d "${INSTALL_DIR}" ]; then
    git clone "${REPO_URL}" "${INSTALL_DIR}"
else
    cd "${INSTALL_DIR}" && git pull origin main
fi

cd "${INSTALL_DIR}"
mkdir -p data logs config

if [ ! -f ".env" ]; then
    cp .env.example .env
    chmod 600 .env
    echo "[+] Created .env from .env.example (insert OKX Demo API keys when ready)"
fi

python3 -m py_compile core/*.py agents/*.py execution/*.py predator_daemon.py
echo "[✓] Syntax & bytecode verification passed."
echo "[✓] Run a live microstructure + LLM predictive scan with:"
echo "    python3 ${INSTALL_DIR}/predator_daemon.py --once --debate"
