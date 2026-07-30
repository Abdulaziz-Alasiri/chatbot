import os
import tempfile
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_groq import ChatGroq

from langchain_classic.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# ==========================================
# 1. إعدادات الصفحة الأساسية
# ==========================================
st.set_page_config(page_title="متجري - خدمة العملاء", page_icon="🛍️", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Cairo', sans-serif; direction: rtl; text-align: right; }
    .client-header { background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); padding: 20px; border-radius: 12px; color: white; margin-bottom: 20px; text-align: center; }
    .admin-header { background: linear-gradient(135deg, #434343 0%, #000000 100%); padding: 20px; border-radius: 12px; color: white; margin-bottom: 20px; text-align: center; }
</style>
""", unsafe_allow_html=True)

# مسار مجلد قاعدة البيانات وملف المفتاح
DB_DIR = "./store_db"
API_KEY_FILE = "./groq_key.txt"

def save_api_key(key):
    with open(API_KEY_FILE, "w", encoding="utf-8") as f:
        f.write(key.strip())

def load_api_key():
    if os.path.exists(API_KEY_FILE):
        with open(API_KEY_FILE, "r", encoding="utf-8") as f:
            return f.read().strip()
    return ""

# ==========================================
# 2. دالة لوحة تحكم الشركة (Admin Panel)
# ==========================================
def admin_page():
    st.markdown('<div class="admin-header"><h1>⚙️ لوحة تحكم الإدارة (سري)</h1></div>', unsafe_allow_html=True)
    
    password = st.text_input("أدخل كلمة مرور الإدارة:", type="password")
    
    if password != "admin123":
        if password: 
            st.error("كلمة المرور خاطئة!")
        return

    st.success("تم تسجيل الدخول بنجاح كمدير للنظام.")
    
    st.markdown("### 🔑 1. إعداد مفتاح Groq API")
    current_key = load_api_key()
    new_api_key = st.text_input("مفتاح Groq API الخاص بالمتجر:", value=current_key, type="password")
    
    if st.button("حفظ مفتاح API"):
        if new_api_key:
            save_api_key(new_api_key)
            st.toast("تم حفظ مفتاح Groq API بنجاح! 🔑", icon="✅")
        else:
            st.warning("يرجى كتابة المفتاح أولاً.")

    st.markdown("---")
    st.markdown("### 📤 2. تحديث بيانات المتجر (الكتالوج)")
    
    uploaded_file = st.file_uploader("ارفع ملف PDF الجديد", type=["pdf"])
    
    if uploaded_file and st.button("تحليل وحفظ في قاعدة البيانات"):
        if not load_api_key():
            st.error("⚠️ يرجى إدخال وحفظ مفتاح Groq API أولاً!")
            return

        with st.spinner("جاري استخراج البيانات وبناء الذكاء الاصطناعي..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name

            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
            os.remove(tmp_path)
            
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
            splits = text_splitter.split_documents(docs)

            embeddings = FastEmbedEmbeddings()
            Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory=DB_DIR)
            
            st.success("✅ تم تحديث بيانات البوت بنجاح!")

# ==========================================
# 3. دالة واجهة العملاء (Client Interface)
# ==========================================
def client_page():
    st.markdown('<div class="client-header"><h1>🛍️ متجري - كيف يمكنني مساعدتك؟</h1></div>', unsafe_allow_html=True)

    groq_api_key = load_api_key()

    if not os.path.exists(DB_DIR) or not groq_api_key:
        st.info("⚠️ الخدمة تحت الصيانة حالياً. سنكون معك قريباً!")
        return

    embeddings = FastEmbedEmbeddings()
    vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    retriever = vectorstore.as_retriever()
    
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.1-8b-instant", temperature=0.2)

    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", "أعد صياغة السؤال الأخير بناءً على تاريخ المحادثة ليكون مفهوماً بذاته. لا تجب عليه."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "أنت مساعد خدمة عملاء لمتجر. أجب بناءً على السياق فقط. إذا لم تجد الإجابة قل 'عذراً لا أملك هذه المعلومة'.\n\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for message in st.session_state.chat_history:
        role = "user" if isinstance(message, HumanMessage) else "assistant"
        avatar = "👤" if role == "user" else "🤖"
        with st.chat_message(role, avatar=avatar):
            st.write(message.content)

    if user_input := st.chat_input("اكتب استفسارك هنا..."):
        with st.chat_message("user", avatar="👤"):
            st.write(user_input)

        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("جاري التفكير..."):
                response = rag_chain.invoke({
                    "input": user_input,
                    "chat_history": st.session_state.chat_history
                })
                answer = response["answer"]
                st.write(answer)

        st.session_state.chat_history.append(HumanMessage(content=user_input))
        st.session_state.chat_history.append(AIMessage(content=answer))

# ==========================================
# 4. التوجيه الخفي عبر الـ Query Parameters
# ==========================================
# قراءة معاملات الرابط
query_params = st.query_params

# إذا احتوى الرابط على admin=true تفتح لوحة الإدارة، غير ذلك تفتح واجهة العملاء
if query_params.get("admin") == "true":
    admin_page()
else:
    client_page()
