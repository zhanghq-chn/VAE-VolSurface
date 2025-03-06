ALTER TABLE fwdprc ADD COLUMN expiration_unix INTEGER;
UPDATE fwdprc 
SET expiration_unix = unixepoch(expiration) 
WHERE expiration IS NOT NULL;

CREATE INDEX idx_fwd_exp_unix on fwdprc (expiration_unix);