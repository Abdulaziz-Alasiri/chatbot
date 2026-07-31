import os
import tempfile
import sqlite3
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
# 1. إعدادات الصفحة والتنسيقات البصرية
# ==========================================
st.set_page_config(page_title="مساعد العود الملكي", page_icon="🪵", layout="wide")

DB_DIR = "./store_db"
API_KEY_FILE = "./groq_key.txt"
ANALYTICS_DB = "./chat_analytics.db"

# إخفاء عناصر Streamlit وإتاحة ثيم العود الملكي
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
    .stApp {
        background-color: #1A120B;
        color: #F5EBE6;
    }
    .stButton>button {
        background: linear-gradient(45deg, #B8860B, #D4AF37) !important;
        color: #1A120B !important;
        font-weight: bold !important;
        border-radius: 8px !important;
        border: none !important;
    }
    .stTextInput input, .stTextArea textarea {
        background-color: #261C14 !important;
        color: #F5EBE6 !important;
        border: 1px solid #D4AF37 !important;
        border-radius: 8px !important;
    }
    [data-testid="stChatMessage"] {
        background-color: #261C14;
        border-radius: 12px;
        border: 1px solid #3D2C1E;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. إدارة قاعدة بيانات التحليلات (SQLite)
# ==========================================
def init_analytics_db():
    conn = sqlite3.connect(ANALYTICS_DB)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS chat_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            user_question TEXT,
            bot_answer TEXT,
            answered_successfully INTEGER
        )
    ''')
    conn.commit()
    conn.close()

def log_chat(question, answer):
    init_analytics_db()
    conn = sqlite3.connect(ANALYTICS_DB)
    c = conn.cursor()
    
    # التأكد إذا كان البوت عجز عن الإجابة
    success = 0 if "عذراً" in answer or "لا أملك" in answer else 1
    
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute('''
        INSERT INTO chat_logs (timestamp, user_question, bot_answer, answered_successfully)
        VALUES (?, ?, ?, ?)
    ''', (now, question, answer, success))
    conn.commit()
    conn.close()

def get_analytics_data():
    init_analytics_db()
    conn = sqlite3.connect(ANALYTICS_DB)
    df = pd.read_sql_query("SELECT * FROM chat_logs ORDER BY id DESC", conn)
    conn.close()
    return df

# ==========================================
# 3. دوال مساعدة لحفظ وتفريغ المفاتيح والملفات
# ==========================================
def save_api_key(key):
    with open(API_KEY_FILE, "w", encoding="utf-8") as f:
        f.write(key.strip())

def load_api_key():
    if os.path.exists(API_KEY_FILE):
        with open(API_KEY_FILE, "r", encoding="utf-8") as f:
            return f.read().strip()
    return ""

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
        elif file_ext in [".xlsx", ".xls", ".csv"]:
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
# 4. لوحة تحكم الإدارة والتحليلات (Admin + Analytics)
# ==========================================
def admin_page():
    st.title("🪵 لوحة تحكم الإدارة والتحليلات")
    st.divider()

    password = st.text_input("أدخل كلمة مرور الإدارة:", type="password")
    if password != "admin123":
        if password: 
            st.error("كلمة المرور خاطئة!")
        return

    st.success("تم تسجيل الدخول بنجاح")

    # تبويبات الإدارة
    tab1, tab2 = st.tabs(["📊 تحليلات المحادثات والعملاء", "⚙️ الإعدادات وتحديث البيانات"])

    # --- التبويب الأول: التحليلات ---
    with tab1:
        st.subheader("📈 نظرة عامة على نشاط البوت")
        df_logs = get_analytics_data()

        if df_logs.empty:
            st.info("لا توجد محادثات مسجلة حتى الآن.")
        else:
            total_chats = len(df_logs)
            successful_chats = len(df_logs[df_logs['answered_successfully'] == 1])
            unanswered_chats = total_chats - successful_chats
            success_rate = int((successful_chats / total_chats) * 100) if total_chats > 0 else 0

            # بطاقات المؤشرات الرئيسية (KPIs)
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("إجمالي المحادثات", total_chats)
            col2.metric("إجابات ناجحة", successful_chats)
            col3.metric("أسئلة غير مجابة", unanswered_chats)
            col4.metric("نسبة النجاح", f"{success_rate}%")

            st.divider()
            
            # عرض الأسئلة غير المجابة كأولوية للتحديث
            st.subheader("⚠️ أسئلة عجز البوت عن إجابتها (حدث الكتالوج بناءً عليها):")
            unanswered_df = df_logs[df_logs['answered_successfully'] == 0]
            if not unanswered_df.empty:
                st.dataframe(unanswered_df[['timestamp', 'user_question']], use_container_width=True)
            else:
                st.success("ما شاء الله! البوت أجاب على كافة الأسئلة بنجاح 🎉")

            st.divider()
            st.subheader("📜 سجل المحادثات الكامل:")
            st.dataframe(df_logs[['timestamp', 'user_question', 'bot_answer']], use_container_width=True)

    # --- التبويب الثاني: الإعدادات والملفات ---
    with tab2:
        st.subheader("🔑 1. مفتاح Groq API")
        current_key = load_api_key()
        new_api_key = st.text_input("مفتاح Groq API:", value=current_key, type="password")
        if st.button("حفظ المفتاح"):
            save_api_key(new_api_key)
            st.toast("تم حفظ مفتاح API بنجاح! 🔑")

        st.divider()
        st.subheader("📤 2. تحديث كتالوج المتجر")
        uploaded_files = st.file_uploader("ارفع الملفات (PDF, Word, Excel, CSV):", type=["pdf", "docx", "xlsx", "csv"], accept_multiple_files=True)
        
        if st.button("بدء المعالجة والتحديث"):
            if not uploaded_files:
                st.warning("رجاءً ارفع ملفاً واحداً على الأقل.")
                return

            all_docs = []
            with st.spinner("جاري قراءة واستخراج البيانات..."):
                for uploaded_file in uploaded_files:
                    docs = load_file_documents(uploaded_file.read(), uploaded_file.name)
                    all_docs.extend(docs)

            if all_docs:
                with st.spinner("جاري تحديث قاعدة البيانات..."):
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
                    splits = text_splitter.split_documents(all_docs)
                    embeddings = FastEmbedEmbeddings()
                    Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory=DB_DIR)
                st.success("✅ تم تحديث قاعدة بيانات العود بنجاح!")

# ==========================================
# 5. واجهة العملاء (Client Interface)
# ==========================================
def client_page():
    st.title("✨ خبير العود الملكي")
    st.markdown("أهلاً بك يا طيب! أنا مستشارك الذكي للإجابة عن أنواع العود، دهن العود، والأسعار.")

    groq_api_key = load_api_key()
    if not groq_api_key or not os.path.exists(DB_DIR):
        st.info("⚠️ المتجر تحت الصيانة حالياً. سنكون معك قريباً!")
        return

    embeddings = FastEmbedEmbeddings()
    vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.1-8b-instant", temperature=0.2)

    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", "أعد صياغة السؤال الأخير بناءً على المحادثة ليكون مفهوماً بذاته بدون الإجابة عليه."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    system_prompt = (
        "أنت 'خبير العود الملكي'، مستشار مبيعات خبير لمتجر عود وعطور فاخرة.\n"
        "أجب بناءً على السياق فقط بأسلوب راقي، محترم، وعربي فصيح وبسيط.\n"
        "رحّب بالعميل بلباقة (مثل: أهلاً بك يا طيب، أنرت متجرنا).\n"
        "إذا لم تجد المعلومة قل بأسلوب لطيف: 'عذراً يا طيب، هذه المعلومة غير متوفرة في الكتالوج حالياً'.\n\n"
        "السياق:\n{context}"
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

    for message in st.session_state.chat_history:
        role = "user" if isinstance(message, HumanMessage) else "assistant"
        avatar = "👤" if role == "user" else "🪵"
        with st.chat_message(role, avatar=avatar):
            st.write(message.content)

    if user_input := st.chat_input("اسأل عن أنواع العود، الدهن، أو الأسعار..."):
        with st.chat_message("user", avatar="👤"):
            st.write(user_input)

        with st.chat_message("assistant", avatar="🪵"):
            with st.spinner("جاري البحث..."):
                response = rag_chain.invoke({
                    "input": user_input,
                    "chat_history": st.session_state.chat_history
                })
                answer = response["answer"]
                st.write(answer)
                
                # 🚀 تسجيل المحادثة في قاعدة بيانات التحليلات
                log_chat(user_input, answer)

        st.session_state.chat_history.append(HumanMessage(content=user_input))
        st.session_state.chat_history.append(AIMessage(content=answer))

# ==========================================
# 6. التوجيه الخفي عبر Query Parameters
# ==========================================
query_params = st.query_params
if query_params.get("admin") == "true":
    admin_page()
else:
    client_page()
