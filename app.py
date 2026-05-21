import streamlit as st
import pandas as pd
from pybaseball import statcast_batter
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

st.set_page_config(page_title="MLB 防守佈陣優化器", layout="wide")
st.title("⚾ MLB 打者防守佈陣優化器")

st.sidebar.header("搜尋條件")
batter_id = st.sidebar.text_input("輸入 MLB 打者 ID (例如大谷 660271)", "660271")
start_date = st.sidebar.date_input("起始日期", pd.to_datetime("2026-04-01"))

if st.sidebar.button("開始分析"):
    with st.spinner('正在從 Statcast 爬取並計算數據中...'):
        data = statcast_batter(start_date, '2026-05-21', player_id=int(batter_id))
        hits = data[data['type'] == 'X'].dropna(subset=['hc_x', 'hc_y'])
        points = hits[['hc_x', 'hc_y']]
        kmeans = KMeans(n_clusters=9, n_init=10).fit(points)
        centroids = kmeans.cluster_centers_

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(hits['hc_x'], hits['hc_y'], alpha=0.1, c='blue', label='擊球落點')
        ax.scatter(centroids[:, 0], centroids[:, 1], c='red', s=300, marker='X', label='建議站位')
        ax.set_xlim(0, 250)
        ax.set_ylim(0, 250)
        ax.set_title(f"打者 ID: {batter_id} 建議防守站位圖")
        st.pyplot(fig)
