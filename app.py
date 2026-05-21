import streamlit as st
import pandas as pd
from pybaseball import statcast_batter
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from PIL import Image
import requests
from io import BytesIO

# 頁面設定
st.set_page_config(page_title="MLB 防守佈陣優化器", layout="wide")
st.title("⚾ MLB 打者防守佈陣優化器")

# 側邊欄
st.sidebar.header("搜尋條件")
batter_id = st.sidebar.text_input("輸入 MLB 打者 ID", "660271")
start_date_obj = st.sidebar.date_input("起始日期", datetime(2026, 4, 1))

if st.sidebar.button("開始分析"):
    with st.spinner('正在從 Statcast 爬取並計算數據中...'):
        try:
            start_date_str = start_date_obj.strftime('%Y-%m-%d')
            data = statcast_batter(start_date_str, '2026-05-21', player_id=int(batter_id))
            hits = data[data['type'] == 'X'].dropna(subset=['hc_x', 'hc_y'])
            
            if len(hits) < 9:
                st.warning("樣本數不足，請嘗試擴大日期範圍。")
            else:
                points = hits[['hc_x', 'hc_y']]
                kmeans = KMeans(n_clusters=9, n_init=10, random_state=42).fit(points)
                centroids = kmeans.cluster_centers_
                
                # 繪圖
                fig, ax = plt.subplots(figsize=(8, 8))
                
                # 使用線上球場圖連結，直接用 URL 讀取，絕對不會報錯
                url = "https://raw.githubusercontent.com/jldbc/baseball-spraychart/master/field.png"
                response = requests.get(url)
                img = Image.open(BytesIO(response.content))
                ax.imshow(img, extent=[0, 250, 0, 250])
                
                # 繪製數據
                ax.scatter(hits['hc_x'], hits['hc_y'], alpha=0.3, c='blue', s=20, label='擊球落點')
                ax.scatter(centroids[:, 0], centroids[:, 1], c='red', s=400, marker='o', edgecolors='white', linewidth=2, label='建議站位')
                
                ax.set_xlim(0, 250)
                ax.set_ylim(0, 250)
                ax.set_title(f"打者 {batter_id} 數據分析佈陣")
                ax.legend(loc='upper right')
                
                st.pyplot(fig)
                st.success("分析完成！")
        except Exception as e:
            st.error(f"系統錯誤：{e}")

st.markdown("---")
st.info("💡 系統已自動載入專業球場背景圖，無需額外上傳檔案。")
