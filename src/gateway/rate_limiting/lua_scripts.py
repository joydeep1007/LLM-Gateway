"""Redis Lua scripts for atomic token-bucket rate-limit admission.

See PRD Section 7.4 for the dual-bucket admission specification and Section 8.3
for the rationale behind the "no TPM refund" design.

Each bucket is stored as a Redis hash with two fields:
    available       - tokens currently available (float, lazily refilled)
    last_refill_ts  - unix timestamp (seconds) the bucket was last refilled

Both scripts use a lazy-refill strategy: the bucket is only persisted when a
request is admitted. A rejected request leaves the stored state untouched, so
the next admission attempt refills from the last successful save.
"""

from __future__ import annotations

# Single-bucket admission: refill, check capacity, and atomically consume tokens.
#
# KEYS[1] = bucket key (Redis hash with fields "available" and "last_refill_ts")
# ARGV[1] = capacity
# ARGV[2] = refill_rate_per_second
# ARGV[3] = tokens_requested
# ARGV[4] = current_timestamp_seconds (float)
#
# Returns {1, remaining} if admitted, consuming tokens_requested from the bucket.
# Returns {0, retry_after_ms} if rejected, leaving the bucket completely untouched.
single_bucket_lua_script = """
-- %.17g preserves full IEEE-754 double round-trip precision; Lua's default
-- tostring()/%.14g formatting truncates high-precision epoch timestamps and
-- can otherwise introduce spurious refill/consumption drift between calls.
local function fmt(n)
    return string.format("%.17g", n)
end

local capacity = tonumber(ARGV[1])
local refill_rate = tonumber(ARGV[2])
local tokens_requested = tonumber(ARGV[3])
local now = tonumber(ARGV[4])

local state = redis.call("HMGET", KEYS[1], "available", "last_refill_ts")
local available = tonumber(state[1])
local last_refill_ts = tonumber(state[2])

if available == nil or last_refill_ts == nil then
    available = capacity
    last_refill_ts = now
end

available = math.min(capacity, available + refill_rate * (now - last_refill_ts))

if available < tokens_requested then
    local retry_after_ms = (tokens_requested - available) / refill_rate * 1000
    return {0, math.ceil(retry_after_ms)}
end

available = available - tokens_requested

redis.call("HSET", KEYS[1], "available", fmt(available), "last_refill_ts", fmt(now))

return {1, fmt(available)}
"""

# Atomic dual-bucket (RPM + TPM) admission. See PRD Section 7.4.
#
# KEYS[1] = gateway:rl:{team_id}:rpm
# KEYS[2] = gateway:rl:{team_id}:tpm
# ARGV[1] = rpm_capacity
# ARGV[2] = rpm_refill_rate_per_second
# ARGV[3] = tpm_capacity
# ARGV[4] = tpm_refill_rate_per_second
# ARGV[5] = tpm_tokens_requested (estimated_input_tokens + max_output_tokens)
# ARGV[6] = current_timestamp_seconds (float)
#
# Returns {1, remaining_rpm, remaining_tpm} if admitted: 1 RPM token and
# tpm_tokens_requested TPM tokens are consumed and both bucket states are
# persisted atomically.
#
# Returns {0, retry_after_ms} if either bucket lacks capacity: NEITHER bucket is
# mutated, so a rejection never partially consumes the other bucket.
#
# IMPORTANT: there is no refund/return path. Once tpm_tokens_requested has been
# consumed here, it is permanently spent for rate-limiting purposes regardless of
# how many tokens the provider actually ends up using (see PRD Section 8.3).
dual_bucket_lua_script = """
-- %.17g preserves full IEEE-754 double round-trip precision; Lua's default
-- tostring()/%.14g formatting truncates high-precision epoch timestamps and
-- can otherwise introduce spurious refill/consumption drift between calls.
local function fmt(n)
    return string.format("%.17g", n)
end

local function load_bucket(key, capacity, refill_rate, now)
    local state = redis.call("HMGET", key, "available", "last_refill_ts")
    local available = tonumber(state[1])
    local last_refill_ts = tonumber(state[2])

    if available == nil or last_refill_ts == nil then
        available = capacity
        last_refill_ts = now
    end

    return math.min(capacity, available + refill_rate * (now - last_refill_ts))
end

local function save_bucket(key, available, now)
    redis.call("HSET", key, "available", fmt(available), "last_refill_ts", fmt(now))
end

local rpm_capacity = tonumber(ARGV[1])
local rpm_refill_rate = tonumber(ARGV[2])
local tpm_capacity = tonumber(ARGV[3])
local tpm_refill_rate = tonumber(ARGV[4])
local tpm_tokens_requested = tonumber(ARGV[5])
local now = tonumber(ARGV[6])

local rpm_available = load_bucket(KEYS[1], rpm_capacity, rpm_refill_rate, now)
local tpm_available = load_bucket(KEYS[2], tpm_capacity, tpm_refill_rate, now)

local rpm_ok = rpm_available >= 1
local tpm_ok = tpm_available >= tpm_tokens_requested

if not rpm_ok or not tpm_ok then
    local rpm_wait = 0
    if not rpm_ok then
        rpm_wait = (1 - rpm_available) / rpm_refill_rate * 1000
    end

    local tpm_wait = 0
    if not tpm_ok then
        tpm_wait = (tpm_tokens_requested - tpm_available) / tpm_refill_rate * 1000
    end

    return {0, math.ceil(math.max(rpm_wait, tpm_wait))}
end

rpm_available = rpm_available - 1
tpm_available = tpm_available - tpm_tokens_requested

save_bucket(KEYS[1], rpm_available, now)
save_bucket(KEYS[2], tpm_available, now)

return {1, fmt(rpm_available), fmt(tpm_available)}
"""
