import streamlit as st
import pandas as pd
from pybaseball import statcast_batter
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import datetime

st.set_page_config(page_title="MLB 防守佈陣優化器", layout="wide")
st.title("⚾ MLB 打者防守佈陣優化器")

batter_id = st.sidebar.text_input("輸入 MLB 打者 ID", "660271")
start_date_obj = st.sidebar.date_input("起始日期", datetime(2026, 4, 1))

if st.sidebar.button("開始分析"):
    with st.spinner('正在計算中...'):
        try:
            data = statcast_batter(start_date_obj.strftime('%Y-%m-%d'), '2026-05-21', player_id=int(batter_id))
            hits = data[data['type'] == 'X'].dropna(subset=['hc_x', 'hc_y'])
            
            if len(hits) >= 9:
                points = hits[['hc_x', 'hc_y']]
                kmeans = KMeans(n_clusters=9, n_init=10, random_state=42).fit(points)
                centroids = kmeans.cluster_centers_
                
                # 繪製球場與數據
                fig, ax = plt.subplots(figsize=(8, 8))
                
                # 繪製球場背景結構 (硬編碼球場圖，不依賴外部圖片)
                ax.add_patch(patches.Wedge((125, 0), 250, 45, 135, color='#f2f2f2')) # 外野
                ax.add_patch(patches.Rectangle((75, 0), 100, 100, color='#e0c9a6'))   # 內野
                
                # 繪製數據點
                ax.scatter(hits['hc_x'], hits['hc_y'], alpha=0.3, c='blue', s=20, label='擊球落點')
                ax.scatter(centroids[:, 0], centroids[:, 1], c='red', s=400, marker='o', edgecolors='white', linewidth=2, label='建議站位')
                
                ax.set_xlim(0, 250)
                ax.set_ylim(0, 250)
                ax.set_title(f"打者 {batter_id} 最佳化防守佈陣")
                ax.legend()
                st.pyplot(fig)
            else:
                st.warning("數據樣本不足。")
        except Exception as e:
            st.error("分析失敗，請檢查網路連線或打者 ID。")
            
