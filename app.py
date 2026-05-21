import streamlit as st
import pandas as pd
from pybaseball import statcast_batter
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
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
            # 確保日期格式為 'YYYY-MM-DD' 字串
            start_date_str = start_date_obj.strftime('%Y-%m-%d')
            end_date_str = '2026-05-21'
            
            # 爬取數據
            data = statcast_batter(start_date_str, end_date_str, player_id=int(batter_id))
            
            # 過濾擊球數據
            hits = data[data['type'] == 'X'].dropna(subset=['hc_x', 'hc_y'])
            
            if len(hits) < 9:
                st.warning("該打者樣本數不足，請嘗試擴大日期範圍。")
            else:
                # AI 分群
                points = hits[['hc_x', 'hc_y']]
                kmeans = KMeans(n_clusters=9, n_init=10, random_state=42).fit(points)
                centroids = kmeans.cluster_centers_
                
                # 繪圖
                fig, ax = plt.subplots(figsize=(8, 8))
                ax.scatter(hits['hc_x'], hits['hc_y'], alpha=0.1, c='blue', label='擊球落點')
                ax.scatter(centroids[:, 0], centroids[:, 1], c='red', s=300, marker='X', label='建議站位')
                ax.set_xlim(0, 250)
                ax.set_ylim(0, 250)
                ax.set_title(f"打者 {batter_id} 建議防守站位圖")
                ax.legend()
                
                st.pyplot(fig)
                st.success("分析完成！")
        
        except Exception as e:
            st.error(f"發生錯誤：{e}")
            st.write("請檢查輸入的 ID 是否正確，或嘗試更換日期。")

st.markdown("---")
st.info("💡 說明：本系統利用 K-Means 演算法，提供數據驅動的防守站位建議。")
