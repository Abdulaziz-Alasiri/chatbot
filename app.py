def init_analytics_db():
    conn = sqlite3.connect(ANALYTICS_DB)
    c = conn.cursor()
    # 1. إنشاء الجدول إن لم يكن موجوداً
    c.execute('''
        CREATE TABLE IF NOT EXISTS chat_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            timestamp TEXT,
            user_question TEXT,
            bot_answer TEXT,
            answered_successfully INTEGER
        )
    ''')
    
    # 2. التأكد من وجود عمود session_id في حال كان الجدول قديم
    c.execute("PRAGMA table_info(chat_logs)")
    columns = [column[1] for column in c.fetchall()]
    if "session_id" not in columns:
        c.execute("ALTER TABLE chat_logs ADD COLUMN session_id TEXT")
        
    conn.commit()
    conn.close()
