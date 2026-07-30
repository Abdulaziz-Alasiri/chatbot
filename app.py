import os
import tempfile
import pandas as pd
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# ---------------------------------------------------------
# 1. إعدادات الصفحة والهوية البصرية للعود والعطور
# ---------------------------------------------------------
st.set_page_config(
    page_title="مساعد العود الملكي | خدمة العملاء",
    page_icon="🪵",
    layout="wide"
)

DB_DIR = "store_db"
KEY_FILE = "groq_key.txt"

# تطبيق تنسيقات CSS الخاصة بثيم العود الملكي
st.markdown("""
<style>
    /* الخلفية الرئيسية */
    .stApp {
        background-color: #1A120B;
        color: #F5EBE6;
    }

    /* شريط الجانب */
    [data-testid="stSidebar"] {
        background-color: #261C14;
        border-left: 1px solid #D4AF37;
    }

    /* أزرار الإرسال والتفاعل */
    .stButton>button {
        background: linear-gradient(45deg, #B8860B, #D4AF37) !important;
        color: #1A120B !important;
        font-weight: bold !important;
        border-radius: 8px !important;
        border: none !important;
    }

    /* مربع إدخال النص */
    .stTextInput input, .stTextArea textarea {
        background-color: #261C14 !important;
        color: #F5EBE6 !important;
        border: 1px solid #D4AF37 !important;
        border-radius: 8px !important;
    }

    /* العناوين والرموز */
    h1, h2, h3 {
        color: #D4AF37 !important;
        font-family: 'Amiri', 'serif', 'Segoe UI';
    }

    /* فقاعات الدردشة */
    [data-testid="stChatMessage"] {
        background-color: #261C14;
        border-radius: 12px;
        border: 1px solid #3D2C1E;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# 2. دوال مساعدة لحفظ وتفريغ البيانات
# ---------------------------------------------------------
def save_groq_key(key):
    with open(KEY_FILE, "w") as f:
        f.write(key.strip())


def load_groq_key():
    if os.path.exists(KEY_FILE):
        with open(KEY_FILE, "r") as f:
            return f.read().strip()
    return ""


def load_file_documents(file_bytes, file_name):
    file_ext = os.path.splitext(file_name)[1].lower()
    docs = []

    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
        tmp_file.write(file_bytes)
        tmp_path = tmp_file.name

    try:
        if file_ext == ".pdf":
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
        elif file_ext in [".docx", ".doc"]:
            loader = Docx2txtLoader(tmp_path)
            docs = loader.load()
        elif file_ext in [".xlsx", ".xls", ".csv"]:
            if file_ext == ".csv":
                df = pd.read_csv(tmp_path)
            else:
                df = pd.read_excel(tmp_path)

            content_list = []
            for idx, row in df.iterrows():
                row_str = " | ".join([f"{col}: {val}" for col, val in row.items() if pd.notna(val)])
                content_list.append(row_str)

            full_text = "\n".join(content_list)
            from langchain_core.documents import Document
            docs = [Document(page_content=full_text, metadata={"source": file_name})]
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    return docs


# ---------------------------------------------------------
# 3. لوحة تحكم الإدارة (خاصة لمتجر العود)
# ---------------------------------------------------------
def admin_page():
    st.title("🪵 لوحة إدارة متجر العود والعطور")
    st.caption("قم بتغذية البوت بكتالوج المنتجات، قائمة الأسعار، ونوتات العطور.")
    st.divider()

    password = st.text_input("كلمة مرور الأدمن:", type="password")
    if password != "admin123":
        if password:
            st.error("كلمة المرور غير صحيحة")
        return

    st.success("تم تسجيل الدخول بنجاح")

    current_key = load_groq_key()
    groq_api_key = st.text_input("مفتاح Groq API:", value=current_key, type="password")
    if st.button("حفظ المفتاح"):
        save_groq_key(groq_api_key)
        st.success("تم حفظ المفتاح بنجاح!")

    st.divider()
    st.subheader("📦 رفع كشوفات المنتجات والأسعار")
    uploaded_files = st.file_uploader(
        "ارفع ملفات (PDF, Word, Excel, CSV) تحتوي على منتجات العود، الدهن، والمسك:",
        type=["pdf", "docx", "xlsx", "csv"],
        accept_multiple_files=True
    )

    if st.button("بدء معالجة وتدريب البوت"):
        if not uploaded_files:
            st.warning("رجاءً قم برفع ملف واحد على الأقل.")
            return

        all_docs = []
        with st.spinner("جاري قراءة واستخراج بيانات العود والمنتجات..."):
            for uploaded_file in uploaded_files:
                file_bytes = uploaded_file.read()
                docs = load_file_documents(file_bytes, uploaded_file.name)
                all_docs.extend(docs)

        if all_docs:
            with st.spinner("جاري بناء قاعدة المعرفة وتحديث البيانات..."):
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
                splits = text_splitter.split_documents(all_docs)
                embeddings = FastEmbedEmbeddings()
                Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory=DB_DIR)
            st.success("✨ تم تحديث قاعدة بيانات العود بنجاح! البوت جاهز لخدمة العملاء.")


# ---------------------------------------------------------
# 4. واجهة العملاء (مساعد العود الملكي)
# ---------------------------------------------------------
def client_page():
    st.title("✨ خبير العود والعطور الفاخرة")
    st.markdown("أهلاً بك في متجرنا! أنا مساعدك الذكي للإجابة عن أنواع العود، الدهن، الأسعار، والتوصيات.")

    groq_api_key = load_groq_key()
    if not groq_api_key:
        st.info("المتجر تحت الصيانة حالياً (لم يتم إعداد المفتاح بعد).")
        return

    if not os.path.exists(DB_DIR):
        st.info("مرحباً بك! يسعدنا خدمتك قريباً فور رفع كتالوج المنتجات.")
        return

    embeddings = FastEmbedEmbeddings()
    vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.1-8b-instant", temperature=0.2)

    # إعادة صياغة السؤال بناءً على الذاكرة
    contextualize_q_system_prompt = (
        "بناءً على سجل المحادثة والسؤال الأخير للعميل، "
        "قم بإعادة صياغة السؤال ليكون مفهوماً بشكل مستقل دون الحاجة للرجوع لسجل المحادثة. "
        "لا تجب على السؤال، فقط أعد صياغته إذا لزم الأمر."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    # توجيه البوت لتقمص شخصية خبير عود راقي
    system_prompt = (
        "أنت 'خبير العود الملكي'، مستشار مبيعات خبير وودود لمتجر عود وعطور فاخرة.\n"
        "استخدم المعلومات المرفقة فقط للإجابة على استفسارات العميل بأسلوب راقي، محترم، وعربي فصيح وبسيط.\n"
        "رحّب بالعميل بلباقة وبكلمات تليق بمتجر عود (مثل: أهلاً بك يا طيب، أنرت متجرنا، طاب يومك).\n"
        "إذا لم تجد تفاصيل المنتج أو السعر في النصوص المرفقة، قل بأسلوب لطيف: 'عذراً يا طيب، هذه المعلومة غير متوفرة في الكتالوج حالياً، يمكنك التواصل مع الدعم الفني للمزيد من التفاصيل.'\n"
        "لا تخترع أسعاراً أو أشكالاً من رأسك أبداً.\n\n"
        "المعلومات المتوفرة من الكتالوج:\n{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    msgs = StreamlitChatMessageHistory(key="mutton_oud_chat")
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        lambda session_id: msgs,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    # عرض الرسائل السابقة
    if len(msgs.messages) == 0:
        msgs.add_ai_message("أهلاً بك يا طيب في متجرنا للعود والعطور الفاخرة 🌿. كيف يمكنني مساعدتك اليوم؟")

    for msg in msgs.messages:
        st.chat_message(msg.type).write(msg.content)

    # استقبال سؤال العميل
    if user_input := st.chat_input("اسأل عن أنواع العود، ثبات العطور، أو الأسعار..."):
        st.chat_message("human").write(user_input)

        with st.chat_message("ai"):
            with st.spinner("جاري البحث في قائمة العطور والأسعار..."):
                response = conversational_rag_chain.invoke(
                    {"input": user_input},
                    config={"configurable": {"session_id": "default_user"}}
                )
                st.write(response["answer"])


# ---------------------------------------------------------
# 5. التوجيه
# ---------------------------------------------------------
query_params = st.query_params
if query_params.get("admin") == "true":
    admin_page()
else:
    client_page()