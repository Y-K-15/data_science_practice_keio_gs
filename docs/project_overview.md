# データサイエンス実践プロジェクト成果物

## 1. 概要
本アプリは、授業「データサイエンス実践」のプロジェクト成果物です。歌詞を感情分布で表現し、曲同士の類似度を計算して「近い曲」を推薦する仕組みを、Streamlit 上で可視化しました。中心曲の近傍をグラフで探索できるため、Connected Papers 風の体験で音楽の探索が行えます。

## 2. 目的
- 歌詞から得られた感情分布を曲レベルに集約する
- cosine 類似度で曲同士の近さを定量化する
- 似た曲の一覧と、近傍グラフの可視化で「探索可能な推薦」を提供する

## 3. データと特徴量
- 入力: `data/split_song_lyrics_with_BERT2_emotions_en_100k.csv`
- 1行 = 歌詞の1行 + GoEmotions 28次元の確率
- 曲キー:
  - `id` があれば最優先で使用
  - なければ `title___artist` を使用

### 感情列（28次元）
`admiration, amusement, anger, annoyance, approval, caring, confusion, curiosity, desire, disappointment, disapproval, disgust, embarrassment, excitement, fear, gratitude, grief, joy, love, nervousness, optimism, pride, realization, relief, remorse, sadness, surprise, neutral`

## 4. アルゴリズム
### 4.1 曲ベクトルの作成
- 1曲内の歌詞行を平均化（mean pooling）
- `artifacts/song_vectors.parquet` に保存

### 4.2 類似度計算（cosine）
- 曲ベクトルを L2 正規化し、cosine 類似度で Top-K を算出
- Top-K はオンデマンド計算（必要に応じてキャッシュ）

### 4.3 類似度の比較方法（サイドバーで切替可能）
- raw cosine（現行）
- cosine without neutral
- mean-centered cosine
- zscore cosine
- weighted pooling (1-neutral)

※ method の説明は UI 右側の「説明を表示/非表示」で確認できます。

## 5. UI/機能
### 5.1 Top-K 類似曲
- 曲を検索・選択し、Top-K 類似曲を表形式で表示
- 各行に「YouTubeで見る」ボタンを追加（検索結果へ遷移）

### 5.2 Connected graph
- 中心曲 + Top-K 近傍をグラフで可視化
- ノードクリックで詳細カードを表示
  - タイトル / アーティスト / YouTubeで見る / 拡張ボタン
- 「拡張する」でそのノードの近傍を追加
- 「Reset graph」で拡張をリセット

## 6. ディレクトリ構成（主要）
```
app/
  streamlit_app.py      # UIと操作
  build_vectors.py      # 曲ベクトル作成
  recommend.py          # cosine類似度計算
  graph.py              # グラフ作成
  components/           # vis-network コンポーネント
artifacts/
  song_vectors.parquet  # 曲ベクトル
  similarity_topk.parquet (任意)
```

## 7. 起動方法
```
docker compose up --build
```
ブラウザで `http://localhost:8501` にアクセス。

## 8. 学び・ポイント
- 感情ベクトルは高次元でも平均化で曲レベル特徴量にできる
- cosine 類似度は高値に偏りやすく、neutral の影響が大きい
- UI で複数の類似度方法を切替えることで、推薦結果の違いを確認できる
- グラフによる近傍探索で、単なる表より直感的に推薦の関係が理解できる

---
このドキュメントはプロジェクト成果物の概要説明として、授業レポートや発表資料に利用できます。
