import os
import shutil
import tempfile
import sqlite3
import uuid
from datetime import datetime
import pandas as pd
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_groq import ChatGroq

from langchain_classic.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# ==========================================
# 1. إعدادات الصفحة والتحقق من معرف الجلسة
# ==========================================
st.set_page_config(page_title="مساعد العود الملكي", page_icon="🪵", layout="wide")

if "user_session_id" not in st.session_state:
    st.session_state.user_session_id = str(uuid.uuid4())[:8]

DB_DIR = "./store_db"
ANALYTICS_DB = "./chat_analytics.db"

st.markdown("""
<style>
    header, [data-testid="stHeader"], footer, [data-testid="stStatusWidget"] {
        display: none !important;
        visibility: hidden !important;
    }
    .main .block-container {
        padding-top: 1rem !important;
        padding-bottom: 2rem !important;
        max-width: 100% !important;
    }
    .stApp { background-color: #1A120B; color: #F5EBE6; direction: rtl; }
    .stButton>button {
        background: linear-gradient(45deg, #B8860B, #D4AF37) !important;
        color: #1A120B !important; font-weight: bold !important;
        border-radius: 8px !important; border: none !important;
    }
    .stTextInput input, .stTextArea textarea {
        background-color: #261C14 !important; color: #F5EBE6 !important;
        border: 1px solid #D4AF37 !important; border-radius: 8px !important;
    }
    [data-testid="stChatMessage"] {
        background-color: #261C14; border-radius: 12px;
        border: 1px solid #3D2C1E; margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. حفظ وتصنيف المحادثات حسب المستخدِم
# ==========================================
def init_analytics_db():
    conn = sqlite3.connect(ANALYTICS_DB)
    c = conn.cursor()
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
    c.execute("PRAGMA table_info(chat_logs)")
    columns = [column[1] for column in c.fetchall()]
    if "session_id" not in columns:
        c.execute("ALTER TABLE chat_logs ADD COLUMN session_id TEXT")
    conn.commit()
    conn.close()

def log_chat(session_id, question, answer):
    init_analytics_db()
    conn = sqlite3.connect(ANALYTICS_DB)
    c = conn.cursor()
    success = 0 if "عذراً" in answer or "لا أملك" in answer or "غير متوفرة" in answer else 1
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute('''
        INSERT INTO chat_logs (session_id, timestamp, user_question, bot_answer, answered_successfully)
        VALUES (?, ?, ?, ?, ?)
    ''', (session_id, now, question, answer, success))
    conn.commit()
    conn.close()

def get_analytics_data():
    init_analytics_db()
    conn = sqlite3.connect(ANALYTICS_DB)
    df = pd.read_sql_query("SELECT * FROM chat_logs ORDER BY id DESC", conn)
    conn.close()
    return df

# ==========================================
# 3. دوال مساعدة لملفات البيانات
# ==========================================
# 💡 تحسين: جلب المفتاح من الأسرار بدلاً من ملف نصي للحماية
def load_api_key():
    return st.secrets.get("GROQ_API_KEY", "")

def load_file_documents(file_bytes, file_name):
    file_ext = os.path.splitext(file_name)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
        tmp_file.write(file_bytes)
        tmp_path = tmp_file.name

    docs = []
    try:
        if file_ext == ".pdf":
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
        elif file_ext in [".docx", ".doc"]:
            loader = Docx2txtLoader(tmp_path)
            docs = loader.load()
        elif file_ext in [".xlsx", ".xls", ".csv", ".txt", ".md"]:
            if file_ext in [".txt", ".md"]:
                with open(tmp_path, "r", encoding="utf-8") as f:
                    text_content = f.read()
                from langchain_core.documents import Document
                docs = [Document(page_content=text_content, metadata={"source": file_name})]
            else:
                df = pd.read_csv(tmp_path) if file_ext == ".csv" else pd.read_excel(tmp_path)
                content_list = []
                for idx, row in df.iterrows():
                    row_str = " | ".join([f"{col}: {val}" for col, val in row.items() if pd.notna(val)])
                    content_list.append(row_str)
                from langchain_core.documents import Document
                docs = [Document(page_content="\n".join(content_list), metadata={"source": file_name})]
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    return docs

# ==========================================
# 4. لوحة الإدارة والتحليلات
# ==========================================
def admin_page():
    st.title("🪵 لوحة التحليلات ومتابعة الزوار (بث مباشر)")
    st.divider()

    # 💡 تحسين: التحقق من كلمة المرور عبر st.secrets بدلاً من النص الصريح
    admin_password = st.secrets.get("ADMIN_PASSWORD", "admin123")
    password = st.text_input("أدخل كلمة مرور الإدارة:", type="password")
    
    if password != admin_password:
        if password: 
            st.error("كلمة المرور خاطئة!")
        return

    st.sidebar.subheader("🔄 إعدادات التحديث")
    auto_refresh = st.sidebar.checkbox("تفعيل التحديث التلقائي اللحظي", value=False)
    refresh_rate = st.sidebar.slider("معدل التحديث (بالثواني):", min_value=3, max_value=60, value=10)

    tab1, tab2 = st.tabs(["📊 جلسات الزوار والمحادثات", "⚙️ الإعدادات وتحديث الكتالوج"])

    with tab1:
        st.subheader("👥 ملخص جلسات زوار المتجر")
        df_logs = get_analytics_data()

        if df_logs.empty:
            st.info("لا توجد محادثات مسجلة حتى الآن.")
        else:
            total_chats = len(df_logs)
            unique_sessions = df_logs['session_id'].dropna().unique()
            col1, col2 = st.columns(2)
            col1.metric("إجمالي الزوار الفريدين", len(unique_sessions))
            col2.metric("إجمالي الأسئلة الموجهة", total_chats)

            st.divider()

            summary_list = []
            for session_id in unique_sessions:
                user_df = df_logs[df_logs['session_id'] == session_id]
                if not user_df.empty:
                    sorted_user_df = user_df.sort_values("id", ascending=True)
                    first_question = sorted_user_df.iloc[0]['user_question']
                    last_time = sorted_user_df.iloc[-1]['timestamp']
                    chat_count = len(sorted_user_df)
                    
                    summary_list.append({
                        "معرّف الزائر (Session ID)": session_id,
                        "أول سؤال للزائر": first_question,
                        "عدد الأسئلة": chat_count,
                        "تاريخ آخر نشاط": last_time
                    })

            summary_df = pd.DataFrame(summary_list)
            st.write("### 📜 1. قائمة جميع الزوار:")
            st.dataframe(summary_df, use_container_width=True)

            st.divider()

            st.subheader("🔍 2. سجل محادثة الزائر المختار (جدول تفصيلي)")
            selected_session = st.selectbox(
                "اختر معرّف الزائر للبدء في استعراض محادثته:",
                options=unique_sessions,
                key="admin_selected_session",
                format_func=lambda x: f"الزائر [{x}] — (عدد الأسئلة: {len(df_logs[df_logs['session_id'] == x])})"
            )

            if selected_session:
                user_chat = df_logs[df_logs['session_id'] == selected_session].sort_values("id", ascending=True)
                chat_table = user_chat[['timestamp', 'user_question', 'bot_answer']].rename(columns={
                    'timestamp': '⏰ الوقت',
                    'user_question': '👤 سؤال الزائر',
                    'bot_answer': '🪵 إجابة البوت'
                })

                st.write(f"#### 💬 المحادثة التفصيلية للزائر `[{selected_session}]`:")
                st.dataframe(chat_table, use_container_width=True, hide_index=True)

    with tab2:
        st.subheader("📤 1. تحديث كتالوج المتجر")
        uploaded_files = st.file_uploader("ارفع الملفات (PDF, Word, Excel, CSV, TXT, MD):", type=["pdf", "docx", "xlsx", "csv", "txt", "md"], accept_multiple_files=True)
        if st.button("بدء المعالجة والتحديث"):
            if not uploaded_files:
                st.warning("رجاءً ارفع ملفاً على الأقل.")
                return

            all_docs = []
            with st.spinner("جاري قراءة الملفات..."):
                for uploaded_file in uploaded_files:
                    docs = load_file_documents(uploaded_file.read(), uploaded_file.name)
                    all_docs.extend(docs)

            if all_docs:
                with st.spinner("جاري بناء قاعدة المعرفة وتحديث الـ Vector Store..."):
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
                    splits = text_splitter.split_documents(all_docs)
                    embeddings = FastEmbedEmbeddings()
                    
                    # 💡 تحسين: استخدام shutil لحذف المجلد بالكامل بشكل نظيف وآمن
                    if os.path.exists(DB_DIR):
                        try:
                            shutil.rmtree(DB_DIR)
                        except Exception as e:
                            st.error(f"حدث خطأ أثناء تنظيف قاعدة البيانات القديمة: {e}")

                    Chroma.from_documents(
                        documents=splits, 
                        embedding=embeddings, 
                        persist_directory=DB_DIR
                    )
                    
                st.success("✅ تم تحديث الكتالوج بنجاح وبأعلى كفاءة!")

    if auto_refresh:
        import time
        time.sleep(refresh_rate)
        st.rerun()
# ==========================================
# 5. واجهة العملاء (المحسّنة مع زر المسح بجوار الإدخال)
# ==========================================
def client_page():
    st.title("✨ خبير العود الملكي")
    st.markdown("أهلاً بك يا طيب! أنا مستشارك الذكي للإجابة عن أنواع العود، الأدهان، العطور، والأسعار.")

    groq_api_key = load_api_key()
    if not groq_api_key or not os.path.exists(DB_DIR):
        st.info("⚠️ المتجر تحت الصيانة حالياً (تأكد من إعداد ملف الكتالوج ومفتاح API). سنكون معك قريباً!")
        return

    embeddings = FastEmbedEmbeddings()
    vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.1-8b-instant", temperature=0.2)

    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", "أعد صياغة السؤال الأخير بناءً على المحادثة ليكون مفهوماً بذاته بدون الإجابة عليه."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    system_prompt = (
        "أنت 'خبير العود الملكي'، مستشار مبيعات خبير لمتجر عود وعطور فاخرة.\n"
        "مهمتك الإجابة بأسلوب فخم، مرحب، وواضح بناءً على الكتالوج المرفق فقط.\n\n"
        "التعليمات:\n"
        "1. اعتمد حصراً على المعلومات المذكورة في 'السياق المرفق' أدناه.\n"
        "2. عند السؤال عن الأسعار أو المنتجات، اذكر التفاصيل المتاحة ودرجات الثبات بوضوح ونظم الإجابة في نقاط.\n"
        "3. إذا كان المنتج أو تفاصيله غير موجودة صراحة في السياق، أجب بلطف بـ: "
        "'عذراً يا طيب، هذه التفاصيل غير متوفرة في الكتالوج حالياً، يمكنك التواصل مع الدعم الفني.'\n\n"
        "السياق المتاح من الكتالوج:\n{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # 1. إنشاء صندوق محادثة قابل للتمرير (Scrollable) بدلاً من الشاشة الكاملة
    chat_container = st.container(height=550)

    # طباعة المحفوظات داخل الصندوق
    with chat_container:
        for message in st.session_state.chat_history:
            role = "user" if isinstance(message, HumanMessage) else "assistant"
            avatar = "👤" if role == "user" else "🪵"
            with st.chat_message(role, avatar=avatar):
                st.write(message.content)

    # 2. وضع زر المسح وحقل الإدخال بجانب بعضهما تحته
    col_input, col_btn = st.columns([9, 1.5], gap="small")
    
    with col_btn:
        # إضافة مسافة صغيرة لمحاذاة الزر مع مربع النص عمودياً
        st.markdown("<div style='margin-top: 3px;'></div>", unsafe_allow_html=True)
        if st.button("🗑️ مسح", use_container_width=True, help="مسح المحادثة الحالية"):
            st.session_state.chat_history = []
            st.rerun()

    with col_input:
        user_input = st.chat_input("اسأل عن أنواع العود، العطور، أو الأسعار...")

    # 3. معالجة الإدخال وتوليد الإجابة
    if user_input:
        with chat_container:
            # عرض سؤال المستخدم فوراً
            with st.chat_message("user", avatar="👤"):
                st.write(user_input)

            # معالجة وعرض إجابة البوت
            with st.chat_message("assistant", avatar="🪵"):
                with st.spinner("جاري البحث في الكتالوج..."):
                    try:
                        response = rag_chain.invoke({
                            "input": user_input,
                            "chat_history": st.session_state.chat_history
                        })
                        answer = response["answer"]
                        st.write(answer)
                        log_chat(st.session_state.user_session_id, user_input, answer)
                        
                        # تحديث المحفوظات بعد نجاح الرد
                        st.session_state.chat_history.append(HumanMessage(content=user_input))
                        st.session_state.chat_history.append(AIMessage(content=answer))
                    except Exception as e:
                        st.error("عذراً، حدث خطأ مؤقت في الاتصال بالخادم. يرجى المحاولة مرة أخرى بعد قليل.")
                        print(f"Error LLM: {e}")
        
        # إعادة التحميل لتحديث الصفحة بشكل نظيف بعد إضافة الرد
        st.rerun()

# ==========================================
# 6. التوجيه
# ==========================================
query_params = st.query_params
if query_params.get("admin") == "true":
    admin_page()
else:
    client_page()
