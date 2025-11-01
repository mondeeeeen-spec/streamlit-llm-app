# app.py
import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage

load_dotenv()
st.set_page_config(page_title="LLMアプリ（LangChain+Streamlit）", page_icon="🤖")

# --- 状態初期化 ---
if "history" not in st.session_state:
    st.session_state.history = []  # [(role, content), ...]

st.title("LLMアプリ（専門家切替）")
st.markdown("- 下のラジオで専門家を選び、テキストを入力して送信してください。")

role_choice = st.radio(
    "専門家の種類を選択：",
    ["製造現場コンサル（A）", "マーケ戦略アドバイザ（B）"],
    index=0,
    key="role_selector",
)

def build_system(role_name: str) -> str:
    if "製造現場" in role_name:
        return "あなたは製造現場の改善コンサルタントです。安全・品質・コスト・納期の観点で簡潔かつ実務的に答えてください。"
    else:
        return "あなたはマーケティング戦略の専門家です。市場分析・訴求・KPI・実行計画の観点で具体的に提案してください。"

# --- これまでの会話を描画（安定のため chat_message を使用）---
for who, content in st.session_state.history:
    with st.chat_message("user" if who == "user" else "assistant"):
        st.markdown(content)

# --- 入力（chat_input は1つの固定ノードで再描画が安定）---
user_text = st.chat_input("質問を入力：")
if user_text:
    # 画面にユーザー発話を追加
    st.session_state.history.append(("user", user_text))
    with st.chat_message("user"):
        st.markdown(user_text)

    # 生成中のプレースホルダ（単一要素に上書きのみ）
    with st.chat_message("assistant"):
        placeholder = st.empty()

        sys_msg = build_system(role_choice)
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

        full = ""
        with st.spinner("生成中..."):
            for chunk in llm.stream([SystemMessage(content=sys_msg),
                                     HumanMessage(content=user_text)]):
                delta = getattr(chunk, "content", None)
                if delta:
                    full += delta
                    placeholder.markdown(full)  # 上書きのみ。empty()で消さない

        # 確定した返答を履歴へ
        st.session_state.history.append(("assistant", full))

# 任意：説明
with st.expander("アプリの説明", expanded=False):
    st.write("・ラジオで専門家ロールを選び、画面下の入力欄から質問してください。")
