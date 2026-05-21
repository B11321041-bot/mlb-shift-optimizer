import streamlit as st
import pandas as pd
from pybaseball import statcast_batter
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from datetime import datetime

# 頁面設定
st.set_page_config(page_title="MLB 防守佈陣優化器", layout="wide")
st.title("⚾ MLB 打者防守佈陣優化器")

# 側邊欄
st.sidebar.header("搜尋條件")
batter_id = st.sidebar.text_input("輸入 MLB 打者 ID (例如大谷 660271)", "660271")
start_date_obj = st.sidebar.date_input("起始日期", datetime(2026, 4, 1))

if st.sidebar.button("開始分析"):
    with st.spinner('正在從 Statcast 爬取並計算數據中...'):
        try:
            # 轉換日期格式
            start_date_str = start_date_obj.strftime('%Y-%m-%d')
            end_date_str = '2026-05-21'
            
            # 1. 爬取數據
            data = statcast_batter(start_date_str, end_date_str, player_id=int(batter_id))
            hits = data[data['type'] == 'X'].dropna(subset=['hc_x', 'hc_y'])
            
            if len(hits) < 9:
                st.warning("樣本數不足，請嘗試擴大日期範圍或更換 ID。")
            else:
                # 2. AI 分群 (K-Means)
                points = hits[['hc_x', 'hc_y']]
                kmeans = KMeans(n_clusters=9, n_init=10, random_state=42).fit(points)
                centroids = kmeans.cluster_centers_
                
                # 3. 繪圖 (不讀取外部圖片，確保穩定性)
                fig, ax = plt.subplots(figsize=(8, 8))
                ax.set_facecolor('#f8f9fa') # 淺色背景
                
                # 繪製擊球落點
                ax.scatter(hits['hc_x'], hits['hc_y'], alpha=0.3, c='blue', s=25, label='擊球落點')
                
                # 繪製建議站位
                ax.scatter(centroids[:, 0], centroids[:, 1], c='red', s=400, marker='o', 
                           edgecolors='white', linewidth=2, label='建議防守位')
                
                ax.set_xlim(0, 250)
                ax.set_ylim(0, 250)
                ax.set_title(f"打者 {batter_id} 數據分析佈陣", fontsize=16)
                ax.legend(loc='upper right')
                
                st.pyplot(fig)
                st.success("分析完成！")
        
        except Exception as e:
            st.error(f"系統錯誤：{e}")
            st.write("請確認 ID 正確，或是該打者在選定期間內沒有足夠的擊球數據。")

st.markdown("---")
st.info("💡 說明：此系統利用 KMeans 聚類演算法，根據 Statcast 擊球落點數據提供客觀的防守站位建議。")
