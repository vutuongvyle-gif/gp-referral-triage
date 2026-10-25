CREATE TABLE IF NOT EXISTS referral_runs (
  run_id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  case_id        TEXT NOT NULL,
  specialty      TEXT NOT NULL,
  model          TEXT,
  config         JSONB,
  recommendation TEXT,
  red_flag       BOOLEAN,
  started_at     TIMESTAMPTZ DEFAULT now(),
  finished_at    TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS gp_decisions (
  id          BIGSERIAL PRIMARY KEY,
  case_id     TEXT NOT NULL,
  ai_decision TEXT,
  gp_action   TEXT,
  overridden  BOOLEAN GENERATED ALWAYS AS (gp_action <> 'approve') STORED,
  reason      TEXT,
  decided_at  TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS decline_reasons (
  id          BIGSERIAL PRIMARY KEY,
  case_id     TEXT NOT NULL,
  specialty   TEXT NOT NULL,
  reason_code TEXT NOT NULL CHECK (reason_code IN
              ('MISSING_INVESTIGATION','BELOW_THRESHOLD','NO_FIRST_LINE','OTHER')),
  detail      TEXT,
  logged_at   TIMESTAMPTZ DEFAULT now()
);

CREATE OR REPLACE VIEW v_decline_summary AS
SELECT specialty, reason_code, count(*) AS n
FROM decline_reasons GROUP BY specialty, reason_code
ORDER BY specialty, n DESC;
CREATE DATABASE hapi;
