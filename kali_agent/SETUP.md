# Kali VM setup

Written for VirtualBox on Windows, assuming you know basic Linux commands and
nothing more. Do these in order; each step is checkable.

---

## 1. Give the VM a second network adapter

The agent must be reachable from Windows but not from the internet. A
host-only adapter does exactly that.

1. Shut the Kali VM down completely.
2. VirtualBox → **File → Tools → Network Manager → Host-only Networks**.
   If the list is empty, click **Create**. You should end up with an adapter
   named `VirtualBox Host-Only Ethernet Adapter` on `192.168.56.1/24`.
3. Select the Kali VM → **Settings → Network**:
   - **Adapter 1**: leave as NAT (this is Kali's internet access, needed for `apt`)
   - **Adapter 2**: tick *Enable*, set **Attached to: Host-only Adapter**
4. Start Kali.

Check it worked:

```bash
ip -4 addr show | grep 192.168.56
```

You should see something like `inet 192.168.56.10/24`. Write that address down —
it goes in your Windows `.env`. If you get nothing, run `sudo dhclient eth1`.

From **Windows** (Command Prompt), confirm you can reach it:

```
ping 192.168.56.10
```

If ping fails, nothing else in this guide will work. Fix it first.

---

## 2. Install the tools

```bash
sudo apt update
sudo apt install -y steghide binwalk foremost exiftool tshark python3-pip python3-venv
```

`tshark` will ask whether non-root users should capture packets. Answer **No** —
this app only reads uploaded PCAP files, it never captures live traffic.

Then the two that aren't in the default repos:

```bash
sudo apt install -y sherlock theharvester
```

If either is missing on your Kali version, install via pipx instead:

```bash
sudo apt install -y pipx
pipx install sherlock-project
pipx install theHarvester
pipx ensurepath
```

`nmap` ships with Kali already. Verify everything:

```bash
for t in steghide binwalk foremost exiftool tshark nmap sherlock theHarvester; do
  command -v $t >/dev/null && echo "OK   $t" || echo "MISS $t"
done
```

Missing tools aren't fatal — the agent reports them as unavailable and the UI
greys out those buttons.

---

## 3. Install the agent

Copy `agent.py` and `requirements.txt` into the VM (shared folder, or just
`git clone` your repo inside Kali).

```bash
mkdir -p ~/dd-agent && cd ~/dd-agent
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## 4. Generate the shared token

```bash
python3 -c "import secrets; print(secrets.token_hex(32))"
```

Copy the output. It goes in **two** places and must match exactly:

- In Kali: `export AGENT_TOKEN=<value>`
- On Windows, in `.env`: `KALI_AGENT_TOKEN=<value>`

To make it stick across reboots in Kali:

```bash
echo 'export AGENT_TOKEN=<value>' >> ~/.bashrc
echo 'export AGENT_HOST=192.168.56.10' >> ~/.bashrc
source ~/.bashrc
```

---

## 5. Run it

```bash
cd ~/dd-agent && source venv/bin/activate
python3 agent.py
```

You should see `Kali agent listening on http://192.168.56.10:7000`.

From **Windows**, in a browser: <http://192.168.56.10:7000/health> →
`{"status":"up",...}`. That's the bridge working.

---

## 6. Snapshot the VM

VirtualBox → **Machine → Take Snapshot**, name it `clean-agent-installed`.

You are going to run carving and extraction tools against files you don't
control. If something goes wrong inside the VM, you roll back in thirty seconds
instead of rebuilding for an afternoon. Do this now, not later.

---

## Running it as a service (optional, later)

Once you're tired of starting it by hand:

```bash
sudo tee /etc/systemd/system/dd-agent.service >/dev/null <<'EOF'
[Unit]
Description=Deepfake Defence Kali agent
After=network.target

[Service]
User=kali
WorkingDirectory=/home/kali/dd-agent
Environment=AGENT_HOST=192.168.56.10
Environment=AGENT_TOKEN=REPLACE_ME
ExecStart=/home/kali/dd-agent/venv/bin/python3 /home/kali/dd-agent/agent.py
Restart=on-failure

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now dd-agent
sudo systemctl status dd-agent
```

---

## Troubleshooting

| Symptom | Cause |
|---|---|
| `Kali VM is not reachable` in the web app | VM off, agent not running, or wrong IP in `.env` |
| `Agent rejected the token` | `AGENT_TOKEN` and `KALI_AGENT_TOKEN` differ |
| `steghide is not installed in this VM` | `sudo apt install steghide` |
| ping works, port 7000 refused | agent bound to `127.0.0.1` — set `AGENT_HOST` |
| `not on the agent's scan allowlist` | working as intended; see `security.py` |
