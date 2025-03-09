import sqlite3

conn = sqlite3.connect('transcripts.db')

# Create source files table first
conn.execute('''
CREATE TABLE IF NOT EXISTS source_files (
    id INTEGER PRIMARY KEY,
    file_name TEXT NOT NULL,
    video_url TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
''')

# Create subtitles table with foreign key relationship
conn.execute('''
CREATE TABLE IF NOT EXISTS transcripts (
    id INTEGER PRIMARY KEY,
    text TEXT NOT NULL,
    start_time TEXT,        -- HH:MM:SS,ms format
    end_time TEXT,
    start_seconds REAL,    -- Decimal seconds for calculations
    end_seconds REAL,
    source_file_id INTEGER NOT NULL,
    embedding BLOB,        -- Serialized vector (pickle/numpy/etc)
    
    FOREIGN KEY (source_file_id) REFERENCES source_files(id),
    CHECK (end_seconds > start_seconds)
);
''')

# Add indexes for common search operations
conn.execute('CREATE INDEX IF NOT EXISTS idx_source_file ON transcripts (source_file_id);')
conn.execute('CREATE INDEX IF NOT EXISTS idx_timestamps ON transcripts (start_seconds);')

conn.commit()
conn.close()