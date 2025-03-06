CREATE TABLE IF NOT EXISTS opprc (
    secid INTEGER,
    date TEXT,
    symbol TEXT,
    symbol_flag INTEGER,
    exdate TEXT,
    last_date TEXT,
    cp_flag TEXT,
    strike_price INTEGER,
    best_bid REAL,
    best_offer REAL,
    volume INTEGER,
    open_interest INTEGER,
    impl_volatility REAL,
    delta REAL,
    gamma REAL,
    vega REAL,
    theta REAL,
    optionid INTEGER,
    cfadj INTEGER,
    am_settlement INTEGER,
    contract_size INTEGER,
    ss_flag INTEGER,
    forward_price REAL,
    expiry_indicator INTEGER,
    root TEXT,
    suffix TEXT,
    cusip TEXT,
    ticker TEXT,
    sic INTEGER,
    index_flag INTEGER,
    exchange TEXT,
    class TEXT,
    issue_type TEXT,
    industry_group TEXT,
    issuer TEXT,
    div_convention TEXT,
    exercise_style TEXT,
    am_set_flag TEXT
);

CREATE TABLE IF NOT EXISTS secprc (
    secid INTEGER,
    date TEXT,
    cusip TEXT,
    ticker TEXT,
    sic INTEGER,
    index_flag INTEGER,
    exchange TEXT,
    class TEXT,
    issue_type TEXT,
    industry_group TEXT,
    low REAL,
    high REAL,
    open REAL,
    close REAL,
    volume INTEGER,
    return REAL,
    cfadj INTEGER,
    shrout INTEGER,
    cfret INTEGER
);

CREATE TABLE IF NOT EXISTS fwdprc (
    secid INTEGER,
    date TEXT,
    expiration TEXT,
    am_settlement INTEGER,
    forward_price REAL,
    cusip TEXT,
    ticker TEXT,
    sic INTEGER,
    index_flag INTEGER,
    exchange TEXT,
    class TEXT,
    issue_type TEXT,
    industry_group TEXT,
    issuer TEXT
);

CREATE TABLE IF NOT EXISTS rate (
    date TEXT,
    days INTEGER,
    rate REAL
);

CREATE INDEX idx_op_date on opprc (date);
CREATE INDEX idx_sec_date on secprc (date);
CREATE INDEX idx_op_strike on opprc (strike_price);

