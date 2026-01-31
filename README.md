# 🚀【SIGNATE】製造業対抗データインサイトチャレンジ2025 解法まとめ

<img src="docs/images/a01.jpg" alt="スピーチ動作4分類" width="480">

〜財務×テキスト×スタッキングによるDX教材購入予測〜<br>
今回はDataiku社協賛で無償でツール使用が可能(但し十分使いこなせなかった)。<br>
目次<br>
1. コンペ概要
2. 私の順位・スコア
3. プログラム解法<br>
3.1. 全体設計<br>
3.2. 欠損値補完（回帰補完）<br>
3.3. 特徴量エンジニアリング<br>
3.4. テキスト特徴量（TF-IDF + SVD）<br>
3.5. モデル構成とスタッキング<br>
3.6. 閾値最適化と提出<br>

## 1. コンペ概要
SIGNATE主催「製造業対抗データインサイトチャレンジ2025」は、
仮想の製造業企業がDX教材を購入するかどうかを予測する二値分類問題です。<br>

**入力データ(説明変数)**
- 財務データ
- 企業属性（業界・上場区分など）
- アンケート
- 企業概要・DX展望などのテキスト

**出力データ(目的変数)**
- 購入フラグ（0 / 1）

**評価指標**：
- F1スコア

単なる数値分類ではなく、
「DXを導入しそうな会社像」をデータから読み取る力が問われるコンペでした。<br>

## 2. 私の順位・スコア
- **Public Score**：0.7434
- **Private Score**：0.7574
- **最終順位**：30位 / 268人

Public から Private へスコアが上がっており、
過学習を抑えた特徴量設計＋CV設計が効いたと考えています。<br>

## 3. プログラム解法
### 3.1 全体設計
全体の流れは以下です。<br>
- 欠損値補完（単純補完は使わない）
- 財務指標の比率・派生特徴量生成
- テキスト情報の数値化（TF-IDF + SVD）
- カテゴリ特徴量のエンコード
- 複数モデルの OOFスタッキングアンサンブル
- PRカーブからF1最適閾値を決定

ポイントは<br>
👉 「特徴量を厚く作り、モデルは平均化する」 です。<br>

### 3.2 欠損値補完（回帰補完）
営業利益と経常利益は、
- 両方欠損する企業は存在しない
- 強い相関がある

という前提がありました。<br>
そこで単純な平均補完ではなく、
回帰モデルによる相互補完を採用しました。<br>
この補完方法はコンペ1位のブリジストンの方から教えて頂きました。
基本に忠実に欠損値補完を実施すべきことが判りました。<br>

```bash
# 営業利益 = a × 経常利益 + b
train_df = train[train["OpProfit"].notna() & train["OrdProfit"].notna()]
lr = LinearRegression()
lr.fit(train_df[["OrdProfit"]], train_df["OpProfit"])

mask = train["OpProfit"].isna()
train.loc[mask, "OpProfit"] = lr.predict(train.loc[mask, ["OrdProfit"]])
```

👉 財務的な整合性を保った補完ができるのが利点です。<br>

### 3.3 特徴量エンジニアリング（財務）
財務系は「絶対値」よりも比率を重視しました。<br>
例：
- 営業利益率
- ROA / ROE
- 自己資本比率
- 売上 / 従業員数

```bash
def safe_div(a, b):
    return np.divide(a, b, out=np.zeros_like(a), where=b!=0)

df["OpMarg"] = safe_div(df["OpProfit"], df["Sales"])
df["ROA"]    = safe_div(df["NetInc"], df["TotAsset"])
df["ROE"]    = safe_div(df["NetInc"], df["Equity"])
```

👉「規模の違う会社を横並びで比較できる」という点で、DX投資判断と相性が良いと考えました。<br>

### 3.4 テキスト特徴量（TF-IDF + SVD）
以下の列はそのままでは使えません。
- 企業概要（Summary）
- 今後のDX展望（DXFutur）

そこで、
- 文字n-gram TF-IDF
- 次元圧縮（TruncatedSVD）
- 正規化

を組み合わせました。<br>

```bash
tfidf = TfidfVectorizer(
    analyzer="char",
    ngram_range=(2,5),
    max_features=4000
)

svd = TruncatedSVD(n_components=32)
pipe = make_pipeline(tfidf, svd, Normalizer())

X_text = pipe.fit_transform(train["Summary"].fillna(""))
```

👉「DX」「デジタル」「改革」などの言葉の温度感をモデルに渡せるのが強みです。<br>

### 3.5 モデル構成とスタッキング
単一モデルではなく、<br>
異なる性質のモデルを組み合わせるスタッキングアンサンブルを採用しました。<br>
1層目に以下のベースモデルを採用し、2層目にロジステック回帰を用いて最終結果を得ました。<br>
- XGBoost
- LightGBM
- CatBoost
- RandomForest
- SVM（RBF）

```bash
oof_meta = np.zeros((X.shape[0], 5))

for fold, (tr, va) in enumerate(skf.split(X, y)):
    model = XGBClassifier(**xgb_params)
    model.fit(X[tr], y[tr])
    oof_meta[va, 0] = model.predict_proba(X[va])[:,1]

メタモデル<br>
Logistic Regression（正則化あり）<br>

meta = LogisticRegression(C=0.5, max_iter=1000)
meta.fit(oof_meta, y)
```

👉ツリー系・距離系・線形系を混ぜることでPrivateスコアの安定性が向上しました。<br>

### 3.6 閾値最適化（F1最大化）
提出前に PRカーブからF1が最大となる閾値を探索します。<br>

```bash
precision, recall, thresholds = precision_recall_curve(y, proba)
f1 = 2 * precision * recall / (precision + recall)
best_thr = thresholds[np.argmax(f1)]
```

👉「0.5固定」はF1ではほぼ最適にならない、というのが実感です。<br>

## おわりに
このコンペを通じて基本に忠実にデータサイエンスすることを改めて学びました。
- 欠損値処理の設計
- 特徴量にビジネス文脈を乗せる重要性
- スタッキング＋閾値最適化の安定感

今後データテーブルの分類コンペに取り組む方の参考になれば幸いです。<br>


## 入力データセット(Input Dataset)

## 動作環境(Execution Environment)
- Windows / Python(Anaconda等)など

## 基本的な使い方(Basic Usage)

## 出力ファイルと保存先(Output Files and Storage)

## フォルダ構成(Folder Structure)

## ファイルサイズ(File Size)

## 関連リンク(Related Links)
SIGNATE 製造業対抗データ分析コンペ<br>
https://prtimes.jp/main/html/rd/p/000000263.000038674.html<br>

## 注意事項(Notes)
None


