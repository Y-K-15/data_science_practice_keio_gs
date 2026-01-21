# data_science_practice_keio_gs

GoEmotions 28次元の感情ベクトルを曲単位で平均し、cosine類似度で類似曲を推薦するStreamlitアプリです。
PyvisでConnected Papers風の近傍グラフも表示します。

## 構成

- `data/split_song_lyrics_with_BERT2_emotions_en_100k.csv`: 入力CSV
- `artifacts/song_vectors.parquet`: 曲単位ベクトル（自動生成）
- `artifacts/similarity_topk.parquet`: 類似度キャッシュ（任意）
- `app/streamlit_app.py`: Streamlit UI
- `app/build_vectors.py`: 曲ベクトル生成
- `app/recommend.py`: 類似度計算
- `app/graph.py`: グラフ描画

## 使い方

1. データ配置
   - `data/split_song_lyrics_with_BERT2_emotions_en_100k.csv` を置きます。

2. 起動

```bash
docker compose up --build
```

3. アクセス
   - ブラウザで `http://localhost:8501` を開きます。

4. 操作
   - 曲を検索して選択
   - Top-K を指定
   - 類似度の計算方法を選択（raw / neutral除外 / mean-center / z-score / weighted）
   - 表とグラフで類似曲を確認
   - グラフ上のノードをクリックして近傍を追加表示
   - Reset graph ボタンでグラフをリセット

## ローカル実行（任意）

Dockerを使わずに実行する場合:

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

## メモ

- 初回起動時に `artifacts/song_vectors.parquet` を生成します。
- サイドバーのボタンから再生成や類似度キャッシュ作成ができます。
