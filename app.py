import streamlit as st
import pandas as pd
from pybaseball import statcast_batter
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# 頁面設定
st.set_page_config(page_title="MLB 防守佈陣優化器", layout="wide")
st.title("⚾ MLB 打者防守佈陣優化器")

# 側邊欄設定
st.sidebar.header("搜尋條件")
batter_id = st.sidebar.text_input("輸入 MLB 打者 ID (例如大谷 660271)", "660271")
start_date_obj = st.sidebar.date_input("起始日期", datetime(2026, 4, 1))

if st.sidebar.button("開始分析"):
    with st.spinner('正在從 Statcast 爬取並計算數據中...'):
        try:
            start_date_str = start_date_obj.strftime('%Y-%m-%d')
            end_date_str = '2026-05-21'
            
            # 1. 爬取數據
            data = statcast_batter(start_date_str, end_date_str, player_id=int(batter_id))
            hits = data[data['type'] == 'X'].dropna(subset=['hc_x', 'hc_y'])
            
            if len(hits) < 9:
                st.warning("該打者樣本數不足，請嘗試擴大日期範圍。")
            else:
                # 2. AI 分群
                points = hits[['hc_x', 'hc_y']]
                kmeans = KMeans(n_clusters=9, n_init=10, random_state=42).fit(points)
                centroids = kmeans.cluster_centers_
                
                # 3. 繪圖與疊加背景
                fig, ax = plt.subplots(figsize=(8, 8))
                
                # 繪製球場背景 (假設你已經上傳了 field.png 到 GitHub 根目錄)
                try:
                    img = plt.imread("field.png")
                    ax.imshow(img, extent=[0, 250, 0, 250])
                except FileNotFoundError:
                    st.info("提示：若要顯示球場背景，請上傳一張名為 field.png 的圖片到 GitHub 倉庫根目錄。")
                
                # 繪製擊球落點與紅點站位
                ax.scatter(hits['hc_x'], hits['hc_y'], alpha=0.2, c='blue', s=20, label='擊球落點')
                ax.scatter(centroids[:, 0], centroids[:, 1], c='red', s=400, marker='o', edgecolors='white', linewidth=2, label='建議站位')
                
                ax.set_xlim(0, 250)
                ax.set_ylim(0, 250)
                ax.set_title(f"打者 {batter_id} 數據分析佈陣")
                ax.legend(loc='upper right')
                
                st.pyplot(fig)
                st.success("分析完成！紅圈即為 AI 推薦的守備站位中心點。")
        
        except Exception as e:
            st.error(f"系統錯誤：{e}")

st.markdown("---")
st.info("💡 說明：此系統根據 Statcast 數據進行 K-Means 聚類分析。若要顯示球場背景，請在 GitHub 上傳 field.png。")
