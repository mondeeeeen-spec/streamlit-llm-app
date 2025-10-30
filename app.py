# app.py
import os
from dotenv import load_dotenv
import streamlit as st

# LangChain
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage

# .env の環境変数を読み込み
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ガード
if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY が設定されていません。.env を確認してください。")
    st.stop()

# Streamlit ページ設定
st.set_page_config(page_title="LLMアプリ（LangChain+Streamlit）", page_icon="🤖")

# --- ヘッダ & 概要 ---
st.title("🤖 LLMアプリ（LangChain + Streamlit）")
st.markdown(
    """
**使い方**  
1. 下のラジオで「専門家の種類」を選びます  
2. 入力欄に質問や文章を入れて **送信** を押します  
3. 回答が下に表示されます（ストリーミングで徐々に表示）
    """
)

# --- モデル準備（Python 3.11 / OpenAI 1.x / LangChain 0.3系） ---
llm = ChatOpenAI(
    model="gpt-4o-mini",           # 授業の指定がなければ軽量でOK。提出先が別モデルを要求するなら合わせる
    temperature=0.4,
    streaming=True,                # ストリーミングON
)

# --- 専門家の定義（A/Bで切替） ---
EXPERT_OPTIONS = ["製造現場コンサル（A）", "マーケ戦略アドバイザ（B）"]
expert = st.radio("専門家の種類を選択：", EXPERT_OPTIONS, horizontal=True)

def build_system_prompt(expert_choice: str) -> str:
    if expert_choice == "製造現場コンサル（A）":
        return (
            "あなたは製造現場の生産性改善コンサルタントです。"
            "現場安全、段取り替え短縮、OEE向上、ムダ取りなど具体策を、"
            "専門用語は噛み砕き、手順・効果・注意点まで簡潔に示してください。"
        )
    else:  # マーケ戦略アドバイザ（B）
        return (
            "あなたはマーケティング戦略アドバイザーです。"
            "STP、4P/4C、ファネル、LTV、クリエイティブ検証等の観点から、"
            "実行可能な打ち手を提示し、根拠とKPIも明記してください。"
        )

# --- 応答生成関数（要件：入力テキスト＆ラジオ選択を引数にとり回答文字列を返す） ---
def generate_answer(user_text: str, expert_choice: str) -> str:
    sys_msg = build_system_prompt(expert_choice)
    messages = [
        SystemMessage(content=sys_msg),
        HumanMessage(content=user_text),
    ]
    # ストリーミングで逐次受信 → 最終テキストも作って返す
    full_text = []
    with st.chat_message("assistant"):
        stream_area = st.empty()
        partial = ""
        for chunk in llm.stream(messages):
            delta = chunk.content or ""
            partial += delta
            # 受信中のテキストを随時描画
            stream_area.markdown(partial)
        full_text = partial
    return full_text

# --- 入力UI ---
user_text = st.text_area("入力テキスト", placeholder="ここに質問や文章を入力…", height=120)

# 会話履歴（簡易）
if "history" not in st.session_state:
    st.session_state["history"] = []

col1, col2 = st.columns([1, 1])
with col1:
    run = st.button("送信", type="primary")
with col2:
    summarize = st.button("100文字以内に要約して")

# --- 実行 ---
if run and user_text.strip():
    st.chat_message("user").markdown(user_text)
    answer = generate_answer(user_text, expert)
    st.session_state["history"].append(("user", user_text))
    st.session_state["history"].append(("assistant", answer))

# 追加要件：会話履歴の“要約”テスト
if summarize:
    hist_text = "\n".join([f"{role}: {text}" for role, text in st.session_state["history"]])
    if not hist_text:
        st.warning("まだ会話履歴がありません。先に質問してください。")
    else:
        prompt = "以下の会話履歴を日本語で100文字以内に要約してください。\n\n" + hist_text
        st.chat_message("user").markdown("100文字以内に要約して")
        _ = generate_answer(prompt, expert)

# --- フッタ ---
st.caption("※APIキーは .env で管理し、GitHubには絶対に含めないでください。")
