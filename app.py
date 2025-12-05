import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# --------------------------------------------------
# 画面設定
# --------------------------------------------------
st.set_page_config(
    page_title="地温予測モデル（Random Forest）",
    layout="wide"
)

# タイトル（少し小さいサイズ）
st.markdown(
    "<h3 style='font-size:20px;'>🌱 地温予測モデルアプリ（Random Forest）信大作成</h3>",
    unsafe_allow_html=True
)

st.write("ST_mean_obs を各種気象変数から予測します。")
st.write("※ 入力CSVは少なくとも `ST_mean_obs` を含み、その他は任意の説明変数を選択できます。")

# --------------------------------------------------
# サイドバー：ファイルアップロード & モデル設定
# --------------------------------------------------
st.sidebar.header("設定")

uploaded_file = st.sidebar.file_uploader(
    "CSVファイルをアップロードしてください",
    type=["csv"]
)

test_size = st.sidebar.slider(
    "テストデータ割合（test_size）", 0.1, 0.5, 0.2, 0.05
)
n_estimators = st.sidebar.slider(
    "決定木の本数（n_estimators）", 100, 1000, 500, 50
)
random_state = st.sidebar.number_input(
    "random_state", value=42, step=1
)

# --------------------------------------------------
# メイン処理：CSV 読み込み
# --------------------------------------------------
if uploaded_file is None:
    st.info("左のサイドバーから CSV ファイルをアップロードしてください。")
    st.stop()

try:
    df = pd.read_csv(uploaded_file)
except Exception as e:
    st.error(f"CSVの読み込みでエラーが発生しました: {e}")
    st.stop()

st.subheader("📄 入力データ（先頭5行）")
st.dataframe(df.head())

# date 列があれば日時型に変換（なくても動く）
if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
else:
    df = df.reset_index(drop=True)

# --------------------------------------------------
# 変数一覧の表示
# --------------------------------------------------
st.subheader("📌 CSV内の変数一覧")
all_columns = df.columns.tolist()
st.write(all_columns)

# 目的変数の存在チェック
target_col = "ST_mean_obs"
if target_col not in all_columns:
    st.error("目的変数 `ST_mean_obs` がCSVに含まれていません。")
    st.stop()

# --------------------------------------------------
# 説明変数の選択
# --------------------------------------------------
st.sidebar.subheader("🔧 モデルに使う説明変数（複数選択可）")

# 説明変数の候補（目的変数以外）
candidate_features = [c for c in all_columns if c != target_col]

# よく使う初期値（存在する列だけをデフォルトに採用）
default_candidates = ["TMP_mean_obs", "GSR", "Tw_mea", "TMP_mea_AMD", "TMP_mean_norm"]
default_features = [c for c in default_candidates if c in candidate_features]

selected_features = st.sidebar.multiselect(
    "説明変数を選択してください（目的変数 ST_mean_obs は除く）",
    options=candidate_features,
    default=default_features
)

if len(selected_features) == 0:
    st.error("少なくとも1つの説明変数を選択してください。")
    st.stop()

# --------------------------------------------------
# ラグ変数（過去データ）の使用 オプション（観測 + AMD 両対応）
# --------------------------------------------------
st.sidebar.subheader("📌 ラグ（過去データ）の使用")

lag_features = []
lag_candidate_cols = candidate_features  # ST_mean_obs を除いた全てをラグ候補に

use_lags = st.sidebar.checkbox(
    "過去3日間のラグ特徴量を追加する",
    value=True
)

if use_lags:
    lag_base_cols = st.sidebar.multiselect(
        "ラグを作成する変数を選んでください（観測 or AMD 両方可）",
        options=lag_candidate_cols,
        default=[c for c in ["TMP_mean_obs", "TMP_mea_AMD", "TMP_mean_norm"] if c in lag_candidate_cols]
    )

    for base_col in lag_base_cols:
        for lag in [1, 2, 3]:
            lag_col = f"{base_col}_lag{lag}"
            df[lag_col] = df[base_col].shift(lag)
            lag_features.append(lag_col)

# 最終的に使う説明変数の一覧
feature_cols = selected_features + lag_features

# --------------------------------------------------
# 学習用データセットの作成
# --------------------------------------------------
df_clean = df.dropna(subset=[target_col] + feature_cols).copy()

st.write(f"有効データ数（ラグと説明変数を考慮後）: **{len(df_clean)} 行**")

if len(df_clean) < 10:
    st.warning("有効データが少なすぎて、安定したモデル学習が難しいかもしれません。")
if len(df_clean) < 5:
    st.error("有効データが 5 行未満のため、学習を中止します。")
    st.stop()

X = df_clean[feature_cols]
y = df_clean[target_col]

# --------------------------------------------------
# 学習・評価
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=int(random_state)
)

rf = RandomForestRegressor(
    n_estimators=int(n_estimators),
    random_state=int(random_state)
)
rf.fit(X_train, y_train)

y_pred_test = rf.predict(X_test)

# RMSE を自前で計算（古い scikit-learn でもOK）
mse = mean_squared_error(y_test, y_pred_test)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred_test)

st.subheader("📈 モデル評価指標（テストデータ）")
col1, col2 = st.columns(2)
with col1:
    st.metric("RMSE", f"{rmse:.3f} ℃")
with col2:
    st.metric("R²", f"{r2:.3f}")

# --------------------------------------------------
# 全データに対する予測値の計算
# --------------------------------------------------
df_clean["ST_mean_pred"] = rf.predict(X)

# --------------------------------------------------
# 特徴量重要度の表示
# --------------------------------------------------
st.subheader("📊 特徴量重要度（Feature Importance）")

importances = rf.feature_importances_
fig_imp, ax_imp = plt.subplots(figsize=(6, 4))
ax_imp.bar(feature_cols, importances)
ax_imp.set_ylabel("Importance")
ax_imp.set_xticklabels(feature_cols, rotation=45, ha="right")
ax_imp.set_title("Feature Importance (Random Forest)")
fig_imp.tight_layout()
st.pyplot(fig_imp)

# --------------------------------------------------
# 予測 vs 実測 の散布図（全データ）
# --------------------------------------------------
st.subheader("🔍 予測値 vs 実測値（全データ）")

fig_scatter, ax_scatter = plt.subplots(figsize=(5, 5))
ax_scatter.scatter(df_clean[target_col], df_clean["ST_mean_pred"])
min_val = min(df_clean[target_col].min(), df_clean["ST_mean_pred"].min())
max_val = max(df_clean[target_col].max(), df_clean["ST_mean_pred"].max())
ax_scatter.plot([min_val, max_val], [min_val, max_val])
ax_scatter.set_xlabel("Observed ST_mean_obs")
ax_scatter.set_ylabel("Predicted ST_mean_pred")
ax_scatter.set_title("Predicted vs Observed (All data)")
fig_scatter.tight_layout()
st.pyplot(fig_scatter)

# --------------------------------------------------
# 時系列折れ線グラフ（観測 vs 予測）
# --------------------------------------------------
st.subheader("📆 時系列プロット（ST_mean_obs vs 予測値）")

if "date" in df_clean.columns:
    fig_ts, ax_ts = plt.subplots(figsize=(10, 4))
    ax_ts.plot(df_clean["date"], df_clean[target_col], label="Observed ST_mean_obs")
    ax_ts.plot(df_clean["date"], df_clean["ST_mean_pred"], label="Predicted ST_mean_pred")
    ax_ts.set_xlabel("Date")
    ax_ts.set_ylabel("ST_mean")
    ax_ts.set_title("Time Series of Observed vs Predicted ST_mean")
    ax_ts.legend()
    fig_ts.autofmt_xdate()
    fig_ts.tight_layout()
    st.pyplot(fig_ts)
else:
    st.info("date 列が存在しないため、時系列グラフは表示できません。")

# --------------------------------------------------
# 予測結果付き CSV ダウンロード
# --------------------------------------------------
st.subheader("💾 予測結果付きデータのダウンロード")

st.write("先頭5行（ST_mean_pred を追加）")
st.dataframe(df_clean.head())

csv_buffer = io.StringIO()
df_clean.to_csv(csv_buffer, index=False)
csv_bytes = csv_buffer.getvalue().encode("utf-8-sig")  # 日本語対応

st.download_button(
    label="📥 ST_mean_pred 付きCSVをダウンロード",
    data=csv_bytes,
    file_name="data_with_ST_mean_pred.csv",
    mime="text/csv"
)

# --------------------------------------------------
# 時系列折れ線グラフ（ST_mean_obs と TMP_mean_obs）
# --------------------------------------------------
st.subheader("📆 時系列プロット（ST_mean_obs と TMP_mean_obs）")

if "date" in df_clean.columns and "TMP_mean_obs" in df_clean.columns:
    fig_ts2, ax_ts2 = plt.subplots(figsize=(10, 4))

    # 観測地温（太線）
    ax_ts2.plot(
        df_clean["date"], df_clean["ST_mean_obs"],
        label="ST_mean_obs (Soil Temp)",
        linewidth=2.5
    )

    # 観測気温（太線）
    ax_ts2.plot(
        df_clean["date"], df_clean["TMP_mean_obs"],
        label="TMP_mean_obs (Air Temp)",
        linewidth=2.5
    )

    ax_ts2.set_xlabel("Date")
    ax_ts2.set_ylabel("Temperature (°C)")
    ax_ts2.set_title("Time Series: ST_mean_obs & TMP_mean_obs")
    ax_ts2.legend()

    fig_ts2.autofmt_xdate()
    fig_ts2.tight_layout()
    st.pyplot(fig_ts2)

else:
    st.info("date または TMP_mean_obs 列が存在しないため、時系列グラフを描画できません。")

