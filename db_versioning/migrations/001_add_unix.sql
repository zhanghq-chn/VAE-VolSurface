-- SELECT 'ALTER TABLE ' || sm.name || ' ADD COLUMN ' || info.name || '_unix INTEGER;
-- UPDATE ' || sm.name || ' 
-- SET ' || info.name || '_unix = unixepoch(' || info.name || ') 
-- WHERE ' || info.name || ' IS NOT NULL;'
-- FROM sqlite_master AS sm, pragma_table_info(sm.name) AS info
-- WHERE sm.type = 'table'
-- AND info.type = 'TEXT' 
-- AND info.name LIKE '%date%';

ALTER TABLE opprc ADD COLUMN date_unix INTEGER;
UPDATE opprc 
SET date_unix = unixepoch(date) 
WHERE date IS NOT NULL;

ALTER TABLE opprc ADD COLUMN exdate_unix INTEGER;
UPDATE opprc 
SET exdate_unix = unixepoch(exdate) 
WHERE exdate IS NOT NULL;

ALTER TABLE opprc ADD COLUMN last_date_unix INTEGER;
UPDATE opprc 
SET last_date_unix = unixepoch(last_date) 
WHERE last_date IS NOT NULL;

ALTER TABLE secprc ADD COLUMN date_unix INTEGER;
UPDATE secprc 
SET date_unix = unixepoch(date) 
WHERE date IS NOT NULL;

ALTER TABLE fwdprc ADD COLUMN date_unix INTEGER;
UPDATE fwdprc 
SET date_unix = unixepoch(date) 
WHERE date IS NOT NULL;

ALTER TABLE rate ADD COLUMN date_unix INTEGER;
UPDATE rate 
SET date_unix = unixepoch(date) 
WHERE date IS NOT NULL;

DROP INDEX IF EXISTS idx_op_date;
DROP INDEX IF EXISTS idx_sec_date;

CREATE INDEX idx_op_date_unix on opprc (date_unix);
CREATE INDEX idx_op_exdate_unix on opprc (exdate_unix);
CREATE INDEX idx_op_last_date_unix on opprc (last_date_unix);
CREATE INDEX idx_sec_date_unix on secprc (date_unix);
CREATE INDEX idx_fwd_date_unix on fwdprc (date_unix);
CREATE INDEX idx_rate_date_unix on rate (date_unix);