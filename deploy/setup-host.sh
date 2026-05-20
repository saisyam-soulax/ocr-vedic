#!/usr/bin/env bash
# One-time host setup for LAN deployment (requires sudo).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "Installing systemd unit..."
sudo cp "$REPO_ROOT/deploy/ocr-vedic.service" /etc/systemd/system/ocr-vedic.service
sudo systemctl daemon-reload
sudo systemctl enable --now ocr-vedic.service

echo "Allowing HTTP on port 80 (ufw)..."
sudo ufw allow 80/tcp comment 'Vedic OCR UI' || true

echo "Status:"
sudo systemctl status ocr-vedic.service --no-pager || true
sudo ufw status || true
echo ""
echo "Share with collaborators: http://$(hostname -I | awk '{print $1}')"
