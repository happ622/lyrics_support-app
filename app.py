
# app.py - 作詞支援アプリ本体
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

st.title("🎵 作詞支援アプリ - 芸術動力学モデル")

# --- 入力パラメータ ---
st.sidebar.header("感情展開の編集")
t_values = np.linspace(0.1, 10, 100)

# ユーザー定義の感情曲線（スライダー）
intro = st.sidebar.slider("序章の強さ", 0.0, 1.0, 0.3)
mid = st.sidebar.slider("中盤の葛藤", 0.0, 1.0, 0.5)
outro = st.sidebar.slider("終盤の感動", 0.0, 1.0, 0.2)

# --- 歌詞構造エディタ ---
st.sidebar.header("歌詞構造入力")
lyrics_intro = st.sidebar.text_area("序章の歌詞", "朝焼けの街を歩く")
lyrics_mid = st.sidebar.text_area("中盤の歌詞", "忘れられない記憶がよみがえる")
lyrics_outro = st.sidebar.text_area("終盤の歌詞", "希望が胸を満たす")

# --- 感情構造g(t)を生成 ---
g = (intro*np.exp(-((t_values-2)**2)/1.5) +
     mid*np.exp(-((t_values-5.5)**2)/1.0) +
     outro*np.exp(-((t_values-8.5)**2)/1.2))

g0 = 0.5 + 0.1*np.sin(t_values / 2.5)  # 平凡な予測

# --- パラメータ ---
n = 2
k = 1.0
a, b, c = 1.0, 0.5, 1.0

dg_dt = np.gradient(g, t_values)
delta_g = g - g0
delta_g[delta_g == 0] = 1e-6

S = (delta_g)**n * t_values - (k * t_values) / (n * delta_g * dg_dt)

# --- テンプレート方程式の解 E(t) を近似 ---
E = np.zeros_like(t_values)
E_dot = np.zeros_like(t_values)
dt = t_values[1] - t_values[0]

for i in range(1, len(t_values)):
    E_ddot = (S[i] + b * E_dot[i-1] - c * E[i-1]**n) / a
    E_dot[i] = E_dot[i-1] + E_ddot * dt
    E[i] = E[i-1] + E_dot[i] * dt

# --- 可視化 ---
st.subheader("感情構成と印象エネルギー")
fig, axs = plt.subplots(3, 1, figsize=(8, 10))

axs[0].plot(t_values, g, label="感情構造 g(t)", color='blue')
axs[0].plot(t_values, g0, label="予測 g₀(t)", linestyle='--', color='gray')
axs[0].legend()
axs[0].set_ylabel("Emotion")
axs[0].grid(True)

axs[1].plot(t_values, S, label="S(t) 刺激関数", color='darkorange')
axs[1].axhline(0, linestyle='--', color='gray')
axs[1].set_ylabel("S(t)")
axs[1].grid(True)

axs[2].plot(t_values, E, label="E(t) 印象エネルギー", color='crimson')
axs[2].set_xlabel("Time")
axs[2].set_ylabel("E(t)")
axs[2].grid(True)

st.pyplot(fig)

# --- 歌詞構造の表示 ---
st.subheader("📝 歌詞構造プレビュー")
st.markdown(f"**[序章]** {lyrics_intro}")
st.markdown(f"**[中盤]** {lyrics_mid}")
st.markdown(f"**[終盤]** {lyrics_outro}")

st.markdown("""
**使い方:**
- 左側のスライダーで感情展開を調整。
- 各セクションに対応する歌詞を記入。
- 印象曲線 E(t) を参考に構成を調整。
""")
