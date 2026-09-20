# Digital Evidence Vault

A chain-of-custody file storage system for digital forensics. Built this to
practice applied crypto and Spring Security beyond the basics — every upload
gets SHA-256 fingerprinted and AES-256-GCM encrypted (per-case envelope
encryption), and every action gets written to a hash-chained audit log, so
tampering with a past log entry breaks the chain and `/api/audit/verify-chain`
can point to exactly where.

Stack: Java 21, Spring Boot 3, Spring Security (stateless JWT + revocation +
rate limiting), JPA/Hibernate, AES-GCM, RSA signatures, TOTP MFA.

## Security/forensics pieces

| Concept | Where |
|---|---|
| SHA-256 fingerprint on upload | `HashService` |
| Per-case envelope encryption (AES-256-GCM) | `EncryptionService`, `CaseService` — each case gets its own DEK, wrapped by a server-wide master key |
| RSA signatures | `SignatureService` — proves who certified a file, independent of the hash |
| Hash-chained audit log | `AuditLogService` — each entry embeds the previous entry's hash |
| Audit chain export | `/api/audit/export` — JSON dump so the chain can be re-verified outside this codebase |
| JWT auth + revocation | `JwtUtil`, `TokenBlacklistService` — per-token `jti` revocation on logout |
| TOTP 2FA | `TotpService`, `AuthService.login()` — optional, recommended for admins |
| Case-level access control | `CaseService.assertAccess()` — investigators see only their own/assigned cases |
| Rate limiting | `RateLimitFilter` — sliding window on login/register/upload/download |
| Per-user storage quota | `EvidenceService.enforceQuota()` |
| BCrypt + lockout | `AuthService` — 5 failed attempts locks the account 15 min |
| Role-based access | `SecurityConfig` — only admins touch the audit trail; public registration can never create an admin (see Bootstrap admin below) |

## Project layout

```
src/main/java/com/evidencevault/
  model/        JPA entities (User, Case, EvidenceFile, AuditLogEntry, RevokedToken)
  repository/   Spring Data repositories
  service/      Business logic (encryption, hashing, audit chain, evidence, TOTP, ClamAV client)
  controller/   REST endpoints
  security/     JWT filter/util + rate limiter
  config/       Spring Security config, admin bootstrap, demo data seeder
  dto/          Request/response records
  exception/    Global error handling
```

## Run locally (no Docker)

Requires JDK 21 and Maven.

```bash
# 1. Generate two secrets
export JWT_SECRET_BASE64=$(openssl rand -base64 32)
export EVIDENCE_MASTER_KEY_BASE64=$(openssl rand -base64 32)

# 2. Required: bootstrap the first admin account (see below)
export BOOTSTRAP_ADMIN_USERNAME=admin
export BOOTSTRAP_ADMIN_EMAIL=admin@example.com
export BOOTSTRAP_ADMIN_PASSWORD=ChangeMe123!

# 3. Run (embedded H2 file DB by default, no Postgres needed)
cd evidence-vault
mvn spring-boot:run
```

App's at `http://localhost:8080`.

## Bootstrap admin

Public registration (`POST /api/auth/register`) always creates an
`INVESTIGATOR`, never `ADMIN`, regardless of what's in the request body.
Previously this wasn't the case, and anyone could self-register as admin,
which defeated the whole point of restricting the audit trail. Now the only
way to get the first admin account is `AdminBootstrapSeeder`, which creates
exactly one on startup if none exists yet, using the `BOOTSTRAP_ADMIN_*` env
vars above. Leave them unset and the app still runs fine, but nobody can hit
admin-only endpoints until you set them and restart (or insert an admin row
directly into the DB).

Once you have an admin, promote others:
```bash
curl -X POST $BASE/api/auth/users/promote -H "Authorization: Bearer $ADMIN_TOKEN" \
  -H "Content-Type: application/json" -d '{"username":"someone"}'
```

## Two-factor auth (TOTP)

```bash
# 1. Start setup (returns a Base32 secret + otpauth:// URI)
curl -X POST $BASE/api/auth/mfa/setup -H "Authorization: Bearer $TOKEN"

# 2. Scan into an authenticator app, confirm with the 6-digit code:
curl -X POST $BASE/api/auth/mfa/confirm -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" -d '{"code":"123456"}'
```
After that, login needs a `totpCode` once the password checks out — API
replies `401 {"error":"MFA_REQUIRED"}` if it's missing, and the frontend uses
that to show the code field.

Confirming setup also hands back ten single-use backup codes (`XXXX-XXXX`),
shown once, stored only as BCrypt hashes:
```bash
# lost your authenticator? log in with backupCode instead of totpCode
curl -s -X POST $BASE/api/auth/login -H "Content-Type: application/json" -d '{
  "username":"jdoe","password":"correcthorsebattery","backupCode":"7F3K-9QXR"
}'
# regenerate the whole set (invalidates old codes)
curl -s -X POST $BASE/api/auth/mfa/backup-codes/regenerate -H "Authorization: Bearer $TOKEN"
```

## Malware scanning (optional, needs external ClamAV)

Off by default (`CLAMAV_ENABLED=false`) since it needs a real `clamd` daemon
running somewhere. `ClamAvScanService` talks to it over TCP using clamd's
INSTREAM protocol directly, no client library.

**Via Docker Compose:**
```bash
docker compose up -d clamav
docker compose logs -f clamav   # wait for "Listening on TCP" - first boot
                                  # pulls ~500MB of virus defs, can take a while
# then set CLAMAV_ENABLED=true in .env and:
docker compose up -d app
```

**On a Linux host directly:**
```bash
sudo apt update && sudo apt install -y clamav clamav-daemon
sudo freshclam
sudo systemctl enable --now clamav-daemon
# clamd defaults to a Unix socket - to reach it remotely, edit
# /etc/clamav/clamd.conf: add "TCPSocket 3310" and "TCPAddr 0.0.0.0", then:
sudo systemctl restart clamav-daemon
echo PING | nc localhost 3310   # should reply PONG
```
Then set `CLAMAV_ENABLED=true`, `CLAMAV_HOST`, `CLAMAV_PORT`.

If scanning's enabled but the daemon's unreachable, uploads fail closed
(rejected, not silently let through) — see `ClamAvScanService`.

## Run with Docker Compose (Postgres + optional ClamAV)

```bash
cp .env.example .env
# fill in JWT_SECRET_BASE64, EVIDENCE_MASTER_KEY_BASE64, BOOTSTRAP_ADMIN_*

docker compose up --build
```

## Try it out (curl)

```bash
BASE=http://localhost:8080

# Register (always comes back INVESTIGATOR, ignoring any "role" field sent)
curl -s -X POST $BASE/api/auth/register -H "Content-Type: application/json" -d '{
  "username":"jdoe","email":"jdoe@example.com","password":"correcthorsebattery"
}'

# Login, grab the token
TOKEN=$(curl -s -X POST $BASE/api/auth/login -H "Content-Type: application/json" -d '{
  "username":"jdoe","password":"correcthorsebattery"
}' | python3 -c "import sys,json;print(json.load(sys.stdin)['token'])")

# Create a case (creator gets access automatically)
CASE_ID=$(curl -s -X POST $BASE/api/cases -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" -d '{
  "caseNumber":"CASE-2026-001","title":"Suspicious login incident","description":"Investigating unauthorized access"
}' | python3 -c "import sys,json;print(json.load(sys.stdin)['id'])")

# Upload evidence (rejected if it'd blow your quota, or if ClamAV flags it)
echo "sample log contents" > sample.txt
EVIDENCE_ID=$(curl -s -X POST $BASE/api/cases/$CASE_ID/evidence -H "Authorization: Bearer $TOKEN" -F "file=@sample.txt" | python3 -c "import sys,json;print(json.load(sys.stdin)['id'])")

# Verify integrity
curl -s -X POST $BASE/api/evidence/$EVIDENCE_ID/verify -H "Authorization: Bearer $TOKEN"

# Download it back
curl -s $BASE/api/evidence/$EVIDENCE_ID/download -H "Authorization: Bearer $TOKEN" -o downloaded.txt

# Grant another investigator access
curl -s -X POST $BASE/api/cases/$CASE_ID/assign -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" -d '{"username":"another_investigator"}'

# Log out (revokes this token right away instead of waiting for expiry)
curl -s -X POST $BASE/api/auth/logout -H "Authorization: Bearer $TOKEN"

# --- Admin actions ---
ADMIN_TOKEN=$(curl -s -X POST $BASE/api/auth/login -H "Content-Type: application/json" -d '{
  "username":"'"$BOOTSTRAP_ADMIN_USERNAME"'","password":"'"$BOOTSTRAP_ADMIN_PASSWORD"'"
}' | python3 -c "import sys,json;print(json.load(sys.stdin)['token'])")

curl -s $BASE/api/audit/chain -H "Authorization: Bearer $ADMIN_TOKEN"
curl -s -X POST $BASE/api/audit/verify-chain -H "Authorization: Bearer $ADMIN_TOKEN"
curl -s $BASE/api/audit/export -H "Authorization: Bearer $ADMIN_TOKEN" -o audit-export.json
```

Want to see tamper detection actually fire? Edit a row's `details` column
directly in the DB, then call `/api/audit/verify-chain` again — it'll report
exactly which sequence number broke. To check the chain without trusting this
app's own verifier, pull `/api/audit/export` and recompute the hashes
yourself: `entryHash[i]` should equal `SHA-256(sequenceNumber + timestamp +
actorUsername + action + caseId + evidenceId + details + previousHash)`, and
`previousHash[i+1]` should equal `entryHash[i]`.

## Configuration reference

Everything below has a sane default except the two encryption secrets and the
bootstrap admin credentials (full list with comments in `.env.example`):

| Env var | Default | Purpose |
|---|---|---|
| `JWT_SECRET_BASE64` | *(required)* | Signing key for JWTs |
| `EVIDENCE_MASTER_KEY_BASE64` | *(required)* | Wraps every case's per-case data key |
| `EVIDENCE_PREVIOUS_MASTER_KEYS_BASE64` | *(unset)* | Comma-separated retired master keys, for key rotation |
| `BOOTSTRAP_ADMIN_USERNAME/_EMAIL/_PASSWORD` | *(unset)* | Creates the first admin on startup |
| `SEED_DEMO_DATA` | `true` | Seeds a demo case + evidence; set `false` for real deployments |
| `EVIDENCE_USER_QUOTA_BYTES` | 5 GB | Per-user upload quota |
| `CLAMAV_ENABLED` / `_HOST` / `_PORT` | `false` / `localhost` / `3310` | Optional malware scanning |
| `RATE_LIMIT_BACKEND` | `memory` | `memory` or `redis` (shared across instances) |
| `REDIS_HOST` / `_PORT` / `_PASSWORD` | `localhost` / `6379` / *(unset)* | Only used when `RATE_LIMIT_BACKEND=redis` |

## Deploying it live (free options)

### Render.com (easiest)

1. Push to GitHub.
2. On Render: **New +** → **Web Service** → connect the repo.
3. Render auto-detects the Dockerfile — set Environment to Docker.
4. Add env vars (Environment tab):
   - `JWT_SECRET_BASE64` / `EVIDENCE_MASTER_KEY_BASE64` = fresh `openssl rand -base64 32` output each
   - `BOOTSTRAP_ADMIN_USERNAME`, `BOOTSTRAP_ADMIN_EMAIL`, `BOOTSTRAP_ADMIN_PASSWORD`
   - `SEED_DEMO_DATA=false`
   - `DATABASE_URL`, `DATABASE_USERNAME`, `DATABASE_PASSWORD`, `DATABASE_DRIVER=org.postgresql.Driver` (spin up a free Render Postgres first, it gives you these)
5. Deploy — you get a public HTTPS URL.

### Railway.app

1. Push to GitHub, then **New Project → Deploy from GitHub repo**.
2. Railway detects the Dockerfile automatically.
3. Add a Postgres plugin — it injects `DATABASE_URL` etc. (may need to map Railway's `PGHOST/PGPORT/...` into the format Spring expects).
4. Add the same env vars as above.
5. Deploy.

### Fly.io

```bash
fly launch          # detects Dockerfile, creates fly.toml
fly secrets set JWT_SECRET_BASE64=$(openssl rand -base64 32)
fly secrets set EVIDENCE_MASTER_KEY_BASE64=$(openssl rand -base64 32)
fly secrets set BOOTSTRAP_ADMIN_USERNAME=admin BOOTSTRAP_ADMIN_EMAIL=admin@example.com BOOTSTRAP_ADMIN_PASSWORD=...
fly postgres create # attach a free Postgres, then fly secrets set DATABASE_URL=...
fly deploy
```

Whichever platform you pick, evidence storage (`EVIDENCE_STORAGE_DIR`) needs a
persistent volume, or uploads vanish on every redeploy. All three support
attaching a disk — mount at `/app/data`, set
`EVIDENCE_STORAGE_DIR=/app/data/evidence-blobs`.

## Case freeze

`POST /api/cases/{caseId}/freeze` (creator or admin only) closes a case and
immediately kills every currently-logged-in session for the creator and any
assigned investigators — including tokens already issued that haven't expired
yet. Works by stamping a `tokensValidAfter` watermark on each affected `User`
row; `JwtAuthFilter` rejects any token issued before that watermark, on top of
the existing per-token blacklist. Logged as `CASE_FROZEN` with the list of
revoked usernames.

```bash
curl -s -X POST $BASE/api/cases/$CASE_ID/freeze -H "Authorization: Bearer $TOKEN"
```

## Structured evidence search

Beyond plain `GET /api/evidence/search?q=...`, there's a filterable version.
All params optional, combine with AND:

```bash
curl -s -G "$BASE/api/evidence/search/advanced" -H "Authorization: Bearer $TOKEN" \
  --data-urlencode "q=malware" \
  --data-urlencode "caseId=$CASE_ID" \
  --data-urlencode "uploadedBy=jdoe" \
  --data-urlencode "status=WITNESSED" \
  --data-urlencode "uploadedAfter=2026-01-01T00:00:00Z" \
  --data-urlencode "uploadedBefore=2026-12-31T23:59:59Z"
```
Still scoped to cases the caller can access, same as plain search.

## Per-case key rotation

`POST /api/cases/{caseId}/rotate-key` (creator or admin) re-wraps a case's DEK
under the current master key without touching the actual evidence files —
just the small wrapped-key blob on the `Case` row changes. Supports a
rotation window via `EVIDENCE_PREVIOUS_MASTER_KEYS_BASE64` (comma-separated
retired keys): `EncryptionService.unwrapKey()` tries the current key first,
falls back through retired ones, so cases don't break the moment you roll
`EVIDENCE_MASTER_KEY_BASE64` — just call `rotate-key` on each case before you
drop the old key for good.

```bash
curl -s -X POST $BASE/api/cases/$CASE_ID/rotate-key -H "Authorization: Bearer $TOKEN"
```

## Distributed rate limiting

`RateLimitFilter` delegates counting to a pluggable `RateLimitStore`. Default
is in-memory, per-instance — no extra infra. Set `RATE_LIMIT_BACKEND=redis`
(plus `REDIS_HOST`/`REDIS_PORT`/`REDIS_PASSWORD`) to switch to
`RedisRateLimitStore`, sharing counts across instances via a fixed-window
`INCR` + `EXPIRE`. If Redis drops while this backend's active, the limiter
fails open (requests pass through) rather than taking the app down —
per-account lockout in `AuthService` is still there as a backstop.

```bash
docker compose up -d redis
export RATE_LIMIT_BACKEND=redis REDIS_HOST=localhost REDIS_PORT=6379
```

## Live audit feed (SSE)

`GET /api/audit/stream` (admin-only) opens an SSE connection and pushes each
new `AuditLogEntry` the moment it's recorded — no polling `/api/audit/chain`
on a timer.

```bash
curl -N -H "Authorization: Bearer $ADMIN_TOKEN" $BASE/api/audit/stream
```
Note: the browser's `EventSource` API can't attach custom headers, so a real
frontend would need a fetch-based SSE client or a short-lived signed URL for
this one endpoint. Also, like the default rate limiter, subscribers are
tracked per-instance — in a multi-instance deployment an admin only sees
events from whichever instance their connection landed on.

## Why CSRF is disabled

Deliberate, not an oversight (see `SecurityConfig`). CSRF exploits a
browser's automatic cookie handling — this API never uses cookies for auth,
every request needs an explicit `Authorization: Bearer <token>` header, which
a forged cross-site request can't attach.
