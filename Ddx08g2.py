# -*- coding: utf-8 -*-
import sys, os, re, struct, binascii
import numpy as np
import shap  # 未使用でもOK（そのまま残し）
# -------------------------------------------------------
# ============================================================
# SIGNATE製造業コンペ2025:
# XGBoost + LightGBM + CatBoost スタッキング フルスクリプト
# - 既存の特徴量生成ロジックをそのまま使用
# - 3モデルのOOF予測をメタ特徴量にしてLGBMでスタッキング
# - early stopping, callbacks, enable_categorical は未使用（互換重視）
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_curve, f1_score
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# -------------------------
# 設定
# -------------------------
### RANDOM_SEED変更でモデルの安定性を確認可能
RANDOM_SEED = 42
# RANDOM_SEED = 100
# RANDOM_SEED = 2024
N_SPLITS = 5

# =========================
# 1) データ読込
# =========================
train = pd.read_csv("./data/train.csv", index_col=0)
test0 = pd.read_csv("./data/test.csv", index_col=0)

# =========================
# 2) 列名マッピング
# =========================
col_map = {
    "企業ID": "CorpID", "企業名": "CorpName", "従業員数": "EmpNum",
    "業界": "IndType", "上場種別": "ListType", "特徴": "DealForm",
    "企業概要": "Summary", "組織図": "OrgChart",
    "事業所数": "OffiNum", "工場数": "FactNum", "店舗数": "ShopNum",
    "資本金": "CapFund", "総資産": "TotAsset", "流動資産": "CurAsset", "固定資産": "FixAsset",
    "負債": "LiabTot", "短期借入金": "StLoan", "長期借入金": "LtLoan",
    "純資産": "NetAsset", "自己資本": "Equity",
    "売上": "Sales", "営業利益": "OpProfit", "経常利益": "OrdProfit",
    "当期純利益": "NetInc", "営業CF": "OpCashF", "減価償却費": "DepExp",
    "運転資本変動": "WkCapChg", "投資CF": "InvCashF",
    "有形固定資産変動": "TangChg", "無形固定資産変動(ソフトウェア関連)": "IntgChg",
    "アンケート１": "Qst1", "アンケート２": "Qst2", "アンケート３": "Qst3",
    "アンケート４": "Qst4", "アンケート５": "Qst5", "アンケート６": "Qst6",
    "アンケート７": "Qst7", "アンケート８": "Qst8", "アンケート９": "Qst9",
    "アンケート１０": "Qst10", "アンケート１１": "Qst11",
    "今後のDX展望": "DXFutur", "購入フラグ": "BuyFlag"
}
train.rename(columns=col_map, inplace=True)
test0.rename(columns=col_map, inplace=True)

# =========================
# 2.5) 欠損値補完
# =========================
train0 = train.copy()
test1 = test0.copy()
# 1. 学習用データ　*********************************
# ***********************************************
# (a)営業利益の補完
train_df = train0[train0["OpProfit"].notna() & train0["OrdProfit"].notna()]
X = train_df[["OrdProfit"]]
y = train_df["OpProfit"]
# 2. 回帰モデル
lr = LinearRegression()
lr.fit(X, y)
a = lr.coef_[0]
b = lr.intercept_
print(f"営業利益 = {a:.4f} × 経常利益 + {b:.4f}")
# 3. 補完対象
mask = train0["OpProfit"].isna() & train0["OrdProfit"].notna()
train.loc[mask, "OpProfit"] = lr.predict(train0.loc[mask, ["OrdProfit"]])
# ***********************************************
# (b)経常利益の補完
train_df = train[train0["OrdProfit"].notna() & train0["OpProfit"].notna()]
X = train_df[["OpProfit"]]
y = train_df["OrdProfit"]
# 2. 回帰モデル
lr2 = LinearRegression()
lr2.fit(X, y)
# 3. 補完対象
mask = train0["OrdProfit"].isna() & train0["OpProfit"].notna()
train.loc[mask, "OrdProfit"] = lr2.predict(train0.loc[mask, ["OpProfit"]])

# 2. テスト用データ　*********************************
# ***********************************************
# (a)営業利益の補完
test_df = test1[test1["OpProfit"].notna() & test1["OrdProfit"].notna()]
X = test_df[["OrdProfit"]]
y = test_df["OpProfit"]
# 2. 回帰モデル
lr3 = LinearRegression()
lr3.fit(X, y)
# 3. 補完対象
mask = test1["OpProfit"].isna() & test1["OrdProfit"].notna()
test0.loc[mask, "OpProfit"] = lr3.predict(test1.loc[mask, ["OrdProfit"]])
# ***********************************************
# (b)経常利益の補完
test_df = test1[test1["OrdProfit"].notna() & test1["OpProfit"].notna()]
X = test_df[["OpProfit"]]
y = test_df["OrdProfit"]
# 2. 回帰モデル
lr4 = LinearRegression()
lr4.fit(X, y)
# 3. 補完対象
mask = test1["OrdProfit"].isna() & test1["OpProfit"].notna()
test0.loc[mask, "OrdProfit"] = lr4.predict(test1.loc[mask, ["OpProfit"]])

# =========================
# 3) ユーティリティ
# =========================
def to_num(df, cols):
    d = df.copy()
    for c in cols:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors='coerce')
    return d

def safe_div(a, b):
    out = np.divide(a, b, out=np.zeros_like(a, dtype=float),
                    where=(b != 0) & np.isfinite(b))
    out[~np.isfinite(out)] = 0.0
    return out

def add_financial_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    num_cols = [
        'Sales','OpProfit','OrdProfit','NetInc','TotAsset','CurAsset','FixAsset',
        'LiabTot','StLoan','LtLoan','Equity','NetAsset','OpCashF','DepExp',
        'WkCapChg','TangChg','IntgChg','EmpNum'
    ]
    d = to_num(d, num_cols)

    # 代表比率
    if {'OpProfit','Sales'}.issubset(d.columns): d['OpMarg']   = safe_div(d['OpProfit'].values, d['Sales'].values)
    if {'Equity','TotAsset'}.issubset(d.columns): d['EqRatio']  = safe_div(d['Equity'].values, d['TotAsset'].values)
    if {'LiabTot','TotAsset'}.issubset(d.columns): d['DebtRt']  = safe_div(d['LiabTot'].values, d['TotAsset'].values)
    if {'NetInc','TotAsset'}.issubset(d.columns): d['ROA']      = safe_div(d['NetInc'].values, d['TotAsset'].values)
    if {'NetInc','Equity'}.issubset(d.columns):   d['ROE']      = safe_div(d['NetInc'].values, d['Equity'].values)
    if {'StLoan','LtLoan','LiabTot'}.issubset(d.columns):
        d['LoanRt'] = safe_div((d['StLoan'].values + d['LtLoan'].values), d['LiabTot'].values)
    if {'CurAsset','TotAsset'}.issubset(d.columns): d['CurRatio'] = safe_div(d['CurAsset'].values, d['TotAsset'].values)
    if {'FixAsset','TotAsset'}.issubset(d.columns): d['FixRatio'] = safe_div(d['FixAsset'].values, d['TotAsset'].values)
    if {'OpCashF','Sales'}.issubset(d.columns):     d['OpCFMarg'] = safe_div(d['OpCashF'].values, d['Sales'].values)

    # 追加群
    if {'OrdProfit','Sales'}.issubset(d.columns): d['OrdMarg'] = safe_div(d['OrdProfit'].values, d['Sales'].values)
    if {'NetInc','Sales'}.issubset(d.columns):     d['NetMarg'] = safe_div(d['NetInc'].values, d['Sales'].values)
    if {'Sales','TotAsset'}.issubset(d.columns):   d['ATurn']   = safe_div(d['Sales'].values, d['TotAsset'].values)
    if {'LiabTot','Equity'}.issubset(d.columns):   d['DebtEquity'] = safe_div(d['LiabTot'].values, d['Equity'].values)
    if {'TotAsset','Equity'}.issubset(d.columns):  d['EqMult']  = safe_div(d['TotAsset'].values, d['Equity'].values)
    if {'CurAsset','LiabTot'}.issubset(d.columns): d['CurToLiab'] = safe_div(d['CurAsset'].values, d['LiabTot'].values)
    if {'OpCashF','LiabTot'}.issubset(d.columns):  d['CFtoDebt']  = safe_div(d['OpCashF'].values, d['LiabTot'].values)
    if {'StLoan','LiabTot'}.issubset(d.columns):   d['StLoanRt']  = safe_div(d['StLoan'].values, d['LiabTot'].values)
    if {'LtLoan','LiabTot'}.issubset(d.columns):   d['LtLoanRt']  = safe_div(d['LtLoan'].values, d['LiabTot'].values)
    if {'Sales','EmpNum'}.issubset(d.columns):     d['SalesPerEmp'] = safe_div(d['Sales'].values, d['EmpNum'].values)
    if {'OpProfit','EmpNum'}.issubset(d.columns):  d['OpPerEmp']    = safe_div(d['OpProfit'].values, d['EmpNum'].values)
    if {'WkCapChg','Sales'}.issubset(d.columns):   d['WkCapToSales'] = safe_div(d['WkCapChg'].values, d['Sales'].values)
    if {'TangChg','Sales'}.issubset(d.columns):    d['TangToSales']  = safe_div(d['TangChg'].values, d['Sales'].values)
    if {'IntgChg','Sales'}.issubset(d.columns):    d['IntgToSales']  = safe_div(d['IntgChg'].values, d['Sales'].values)

    # # # ---⓵ 成長性・ギャップ系 ---
    # if {'OrdProfit','Sales','OpProfit'}.issubset(d.columns):
    #     d['ProfitGrowth'] = safe_div(d['OrdProfit'], d['Sales']) - safe_div(d['OpProfit'], d['Sales'])
    # if {'ROE','ROA'}.issubset(d.columns):
    #     d['ROEminusROA'] = d['ROE'] - d['ROA']
    # if {'TotAsset','FixAsset'}.issubset(d.columns):
    #     d['AssetGrowth'] = safe_div((d['TotAsset'] - d['FixAsset']), d['TotAsset'])
    # if {'TotAsset','Equity','NetInc'}.issubset(d.columns):
    #     d['LeverageEffect'] = safe_div(d['TotAsset'], d['Equity']) * safe_div(d['NetInc'], d['TotAsset'])
    # if {'OpCashF','Sales','OpProfit'}.issubset(d.columns):
    #     d['CFMarginGap'] = safe_div(d['OpCashF'], d['Sales']) - safe_div(d['OpProfit'], d['Sales'])
    # if {'Sales','CapFund'}.issubset(d.columns):
    #     d['SalesToCap'] = safe_div(d['Sales'], d['CapFund'])
    # if {'OpCashF','TangChg','IntgChg'}.issubset(d.columns):
    #     d['OpCFToCapex'] = safe_div(d['OpCashF'], (d['TangChg'] + d['IntgChg']))
    # # # ---⓶ 安定性・安全性 ---
    # if {'LiabTot','OpCashF'}.issubset(d.columns):
    #     d['DebtToCF'] = safe_div(d['LiabTot'], d['OpCashF'])
    #     d['OpCFToDebt'] = safe_div(d['OpCashF'], d['LiabTot'])
    # if {'IntgChg','TangChg'}.issubset(d.columns):
    #     d['IntgRatio'] = safe_div(d['IntgChg'], (d['TangChg'] + d['IntgChg']))
    # if {'OpProfit','NetInc'}.issubset(d.columns):
    #     d['OpToNetInc'] = safe_div(d['OpProfit'], d['NetInc'])
    # if {'TangChg','FixAsset'}.issubset(d.columns):
    #     d['TangToFix'] = safe_div(d['TangChg'], d['FixAsset'])
    # # # ---⓷ 効率性・生産性 ---
    # if {'TotAsset','EmpNum'}.issubset(d.columns): d['AssetPerEmp'] = safe_div(d['TotAsset'], d['EmpNum'])
    # if {'OpCashF','EmpNum'}.issubset(d.columns): d['CFPerEmp'] = safe_div(d['OpCashF'], d['EmpNum'])
    # if {'NetInc','EmpNum'}.issubset(d.columns): d['ProfitPerEmp'] = safe_div(d['NetInc'], d['EmpNum'])
    # if {'LiabTot','EmpNum'}.issubset(d.columns): d['LiabPerEmp'] = safe_div(d['LiabTot'], d['EmpNum'])
    # # # ---⓸　複合評価スコア（統合指標） ---
    # if {'ROA','EqRatio','DebtRt','OpCFMarg'}.issubset(d.columns):
    #     d['FinHealthScore'] = (d['ROA'] + d['EqRatio'] - d['DebtRt'] + d['OpCFMarg']) / 4
    # if {'OpMarg','NetMarg','ROE'}.issubset(d.columns):
    #     d['ProfitabilityMix'] = (d['OpMarg'] + d['NetMarg'] + d['ROE']) / 3

    # アンケート要約
    q_cols = [c for c in d.columns if c.startswith('Qst')]
    if q_cols:
        d['QstMean'] = d[q_cols].mean(axis=1)
        d['QstSum']  = d[q_cols].sum(axis=1)
        d['QstStd']  = d[q_cols].std(axis=1).fillna(0.0)
    return d

def log1p_cols(df: pd.DataFrame, cols):
    d = df.copy()
    for c in cols:
        if c in d.columns:
            d[c + '_log'] = np.log1p(pd.to_numeric(d[c], errors='coerce').clip(lower=0))
    return d

def target_encode_oof(train_df, test_df, y, cat_cols, n_splits=5, seed=42, smooth=20):
    """OOF ターゲットエンコード（リーク防止＋スムージング）"""
    tr = train_df.copy(); te = test_df.copy()
    for c in cat_cols:
        if c in tr.columns: tr[c] = tr[c].astype(str).fillna("__NA__")
        if c in te.columns: te[c] = te[c].astype(str).fillna("__NA__")
    if not isinstance(y, pd.Series):
        y = pd.Series(y, name="BuyFlag")
    y = y.reset_index(drop=True)
    global_mean = float(y.mean())

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for col in cat_cols:
        if col not in tr.columns:
            continue
        tr[col + "_TE"] = np.nan
        for tr_idx, va_idx in skf.split(tr, y):
            tmp = pd.DataFrame({col: tr.iloc[tr_idx][col].values,
                                'y_tmp': y.iloc[tr_idx].values})
            grp = tmp.groupby(col)['y_tmp'].agg(['sum', 'count'])
            stats = (grp['sum'] + global_mean * smooth) / (grp['count'] + smooth)
            tr.loc[tr.index[va_idx], col + "_TE"] = (
                tr.iloc[va_idx][col].map(stats).fillna(global_mean).values
            )

        tmp_full = pd.DataFrame({col: tr[col].values, 'y_tmp': y.values})
        grp_full = tmp_full.groupby(col)['y_tmp'].agg(['sum', 'count'])
        stats_full = (grp_full['sum'] + global_mean * smooth) / (grp_full['count'] + smooth)
        te[col + "_TE"] = te[col].map(stats_full).fillna(global_mean).values

        tr[col + "_TE"] = tr[col + "_TE"].astype(float)
        te[col + "_TE"] = te[col + "_TE"].astype(float)
    return tr, te

def tfidf_svd_features(tr_series, te_series, prefix, n_comp=32, max_feat=4000, seed=42):
    """文字n-gramのTF-IDF→SVD（Normalizer付き）"""
    tr_text = tr_series.fillna("").astype(str)
    te_text = te_series.fillna("").astype(str)
    tfidf = TfidfVectorizer(analyzer='char', ngram_range=(2,5),
                            max_features=max_feat, min_df=2)
    svd = TruncatedSVD(n_components=n_comp, random_state=seed)
    norm = Normalizer(copy=False)
    pipe = make_pipeline(tfidf, svd, norm)
    tr_mat = pipe.fit_transform(tr_text)
    te_mat = pipe.transform(te_text)
    tr_df = pd.DataFrame(tr_mat, index=tr_series.index,
                         columns=[f"{prefix}SV{i+1}" for i in range(n_comp)])
    te_df = pd.DataFrame(te_mat, index=te_series.index,
                         columns=[f"{prefix}SV{i+1}" for i in range(n_comp)])
    return tr_df, te_df

# =========================
# 4) 特徴作成
# =========================
y = train['BuyFlag'].astype(int)
X0 = train.drop('BuyFlag', axis=1)

# 比率・派生
X0  = add_financial_features(X0)
test0 = add_financial_features(test0)

# TF-IDF→SVD（Summary / DXFutur）
if 'Summary' in X0.columns and 'Summary' in test0.columns:
    sum_tr, sum_te = tfidf_svd_features(
        X0['Summary'], test0['Summary'],
        prefix="Sum", n_comp=32, max_feat=4000, seed=RANDOM_SEED
    )
    X0 = pd.concat([X0, sum_tr], axis=1)
    test0 = pd.concat([test0, sum_te], axis=1)

if 'DXFutur' in X0.columns and 'DXFutur' in test0.columns:
    fut_tr, fut_te = tfidf_svd_features(
        X0['DXFutur'], test0['DXFutur'],
        prefix="Fut", n_comp=32, max_feat=4000, seed=RANDOM_SEED
    )
    X0 = pd.concat([X0, fut_tr], axis=1)
    test0 = pd.concat([test0, fut_te], axis=1)

# 組織図からフラグ抽出（DX_flag / DXB_flag）
dx_depts = ["DX推進"]
dx_labels = ["DX_flag"]
###
# dx_depts = [
#     "DX推進", "研究開発",
#     "製品開発部", "運用部", "品質保証部", "製造技術部",
#     "生産部", "戦略企画課", "戦略企画部",
#     "技術開発部", "技術本部", "カスタマーサポート部"
# ]
# dx_labels = [
#     "DX_flag", "RD_flag",
#     "PM_flag", "OP_flag", "QA_flag", "MT_flag",
#     "PD_flag", "SP_flag", "SS_flag",
#     "TD_flag", "TH_flag", "CS_flag"
# ]
###
# train側 DX_flag
for j in range(len(dx_depts)):
    X0[dx_labels[j]] = X0["OrgChart"].astype(str).str.contains(dx_depts[j], na=False).astype(int)

# test側 DX_flag（インデックス扱いを安全に修正）
for j in range(len(dx_depts)):
    test0[dx_labels[j]] = test0["OrgChart"].astype(str).str.contains(dx_depts[j], na=False).astype(int)

# DXB_flag（DX推進 + 高DX部門）
high_dx_depts = [
    "製品開発部", "運用部", "品質保証部", "製造技術部",
    "生産部", "戦略企画課", "戦略企画部",
    "技術開発部", "技術本部", "カスタマーサポート部"
]
pattern = "|".join(high_dx_depts)

cond_dx_tr  = X0["OrgChart"].astype(str).str.contains("DX推進", na=False)
cond_high_tr = X0["OrgChart"].astype(str).str.contains(pattern, na=False)
X0["DXB_flag"] = ((cond_dx_tr) & (cond_high_tr)).astype(int)

cond_dx_te  = test0["OrgChart"].astype(str).str.contains("DX推進", na=False)
cond_high_te = test0["OrgChart"].astype(str).str.contains(pattern, na=False)
test0["DXB_flag"] = ((cond_dx_te) & (cond_high_te)).astype(int)

### -----------------------------------
##　非線形パラメータ
# cparas = [
#     "EmpNum"
# ]
cparas = [
    "EmpNum", "OffiNum", "FactNum", "ShopNum", "CapFund", "TotAsset",
    "CurAsset", "FixAsset", "LiabTot", "StLoan", "LtLoan","Sales",
    "NetInc", "OpCashF", "WkCapChg", "InvCashF", "TangChg"
]
###　非線形パラメータのうち相関係数が高いもの(0.1以上)
# cparas = [
#     "EmpNum", "OffiNum", "TotAsset", "CurAsset", "FixAsset", "StLoan",
#     "LtLoan", "NetInc", "OpCashF", "WkCapChg", "InvCashF"
# ]
###　非線形パラメータのうち相関係数が高いもの(0.15以上)
# cparas = [
#     "EmpNum", "StLoan", "FixAsset", "NetInc", "OpCashF", "InvCashF"
# ]
###　非線形パラメータのうち相関係数が低いもの(0.1以下)
# cparas = [
#     "FactNum", "ShopNum", "CapFund", "LiabTot", "Sales", "TangChg"
# ]
for k in range(len(cparas)):
    y1 = pd.DataFrame(y)
    X2 = pd.concat([y1, X0], axis=1)
    test2 = test0.copy()
    assert cparas[k] in X2.columns and "BuyFlag" in X2.columns
    # === 1) X2 からビン境界を学習（q分位, 重複はdrop） ===
    q = 5
    # qcutで境界だけ取得
    _, bins = pd.qcut(X2[cparas[k]], q=q, duplicates="drop", retbins=True)
    # 端を広げて外れ値にも確実にマッチさせる
    bins[0]  = -np.inf
    bins[-1] =  np.inf
    # === 2) 固定境界で X2/test2 を同じ pd.cut でビン化 ===
    X2_bins = pd.cut(X2[cparas[k]], bins=bins, include_lowest=True, ordered=True)
    test2_bins  = pd.cut(test2[cparas[k]],  bins=bins, include_lowest=True, ordered=True)
    # === 3) X2 で各ビンの購入率 → 高い順にスコア(5→1) ===
    rate = X2.groupby(X2_bins)["BuyFlag"].mean().sort_values(ascending=False)
    # スコア配列（例：ビンがk個なら k,k-1,...,1）
    scores_desc = list(range(len(rate), 0, -1))
    rank_map = dict(zip(rate.index.tolist(), scores_desc))  # {Interval: score}
    # === 4) X2/test2 にスコアを付与（Intervalで map できる） ===
    X2[cparas[k]+"_bin"]   = X2_bins.astype(str)            # 解析用にラベルも残す
    X2[cparas[k]+"_score"] = X2_bins.map(rank_map).astype(float)
    X0[cparas[k]+"_score"] = X2_bins.map(rank_map).astype(float)
    test2[cparas[k]+"_bin"]    = test2_bins.astype(str)
    test2[cparas[k]+"_score"]  = test2_bins.map(rank_map).astype(float)
    test0[cparas[k]+"_score"]  = test2_bins.map(rank_map).astype(float)
    # # 欠損（EmpNumがNaN）の扱い：中央スコアで埋める例
    mid_score = int(np.ceil(len(scores_desc)/2)) if len(scores_desc)>0 else 3
    X2[cparas[k]+"_score"] = X2[cparas[k]+"_score"].fillna(mid_score)
    test2[cparas[k]+"_score"]  = test2[cparas[k]+"_score"].fillna(mid_score)
    X0[cparas[k]+"_score"] = X0[cparas[k]+"_score"].fillna(mid_score)
    test0[cparas[k]+"_score"]  = test0[cparas[k]+"_score"].fillna(mid_score)
# # -------------------------------------------------

# 対数変換
log_cols = [
    'CapFund','Sales','TotAsset','CurAsset','FixAsset','LiabTot','StLoan','LtLoan',
    'NetAsset','Equity','OpProfit','OrdProfit','NetInc','OpCashF','DepExp',
    'TangChg','IntgChg'
]
X0   = log1p_cols(X0, log_cols)
test0 = log1p_cols(test0, log_cols)

# 頻度エンコード
for col in ['ListType','DealForm']:
    if (col in X0.columns) and (col in test0.columns):
        freq = X0[col].astype(str).value_counts(normalize=True)
        X0[col + '_Freq']   = X0[col].astype(str).map(freq).fillna(0.0)
        test0[col + '_Freq'] = test0[col].astype(str).map(freq).fillna(0.0)

# OOF ターゲットエンコード
candidate_cats = ['IndType','ListType','DealForm']
cat_cols = [c for c in candidate_cats if c in X0.columns and c in test0.columns]
te_train, te_test = target_encode_oof(
    train_df=X0[cat_cols].copy(),
    test_df=test0[cat_cols].copy(),
    y=y,
    cat_cols=cat_cols,
    n_splits=N_SPLITS,
    seed=RANDOM_SEED,
    smooth=20
)
for c in cat_cols:
    X0[c + '_TE']    = te_train[c + '_TE'].values
    test0[c + '_TE'] = te_test[c + '_TE'].values

# アンケートPCA
q_cols = [c for c in X0.columns if c.startswith('Qst')]
if q_cols:
    q_train = X0[q_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0)
    q_test  = test0[q_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0)
    pca = PCA(n_components=min(3, len(q_cols)), random_state=RANDOM_SEED)
    q_train_pca = pca.fit_transform(q_train)
    q_test_pca  = pca.transform(q_test)
    for i in range(q_train_pca.shape[1]):
        X0[f'QstPCA{i+1}']   = q_train_pca[:, i]
        test0[f'QstPCA{i+1}'] = q_test_pca[:, i]

# 学習に不要な原文テキスト列を除外（生カテゴリも落とす）
drop_text_cols = ['CorpName','Summary','OrgChart','DXFutur','IndType','ListType','DealForm']
X = X0.drop(columns=[c for c in drop_text_cols if c in X0.columns])
test = test0.drop(columns=[c for c in drop_text_cols if c in test0.columns])

# 欠損/無限
X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
test = test.replace([np.inf, -np.inf], np.nan).fillna(0)

# =========================
# 5) スタッキング用モデル定義（XGB + LGB + CAT + RF + SVM）
# =========================
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_curve, f1_score
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd

RANDOM_SEED = 42
N_SPLITS = 5

# クラス不均衡対策（簡易weight）
pos_ratio = y.mean()
neg_ratio = 1 - pos_ratio
scale_pos_weight = neg_ratio / max(pos_ratio, 1e-6)

# --- XGBoost（既存チューニング） ---
xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'learning_rate': 0.025,
    'max_depth': 8,
    'min_child_weight': 2.0,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'n_estimators': 3600,
    'tree_method': 'hist',       # GPUなら 'gpu_hist'
    'scale_pos_weight': scale_pos_weight,
    'random_state': RANDOM_SEED,
}

# --- LightGBM ---
lgb_params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'learning_rate': 0.1,
    'num_leaves': 31,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'min_data_in_leaf': 30,
    'n_estimators': 800,
    'verbose': -1,
    # 'device': 'gpu',
}

# --- CatBoost ---
cat_params = {
    'loss_function': 'Logloss',
    'eval_metric': 'Logloss',
    'learning_rate': 0.07,
    'depth': 4,
    'l2_leaf_reg': 7.0,
    'bootstrap_type': 'Bayesian',
    'bagging_temperature': 1.0,
    'border_count': 254,
    'random_seed': RANDOM_SEED,
    'iterations': 800,
    'verbose': False,
    'allow_writing_files': False,
    # 'task_type': 'GPU', 'devices': '0',
}

# --- RandomForest ---
rf_params = dict(
    n_estimators=2000,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features="sqrt",
    n_jobs=-1,
    random_state=RANDOM_SEED,
    class_weight="balanced_subsample"
)

# --- SVM（第5モデル）---
# RBFカーネル + class_weight=balanced + スケーリング込み
svm_params = dict(
    C=1.0,
    kernel="rbf",
    gamma="scale",
    class_weight="balanced",
    probability=True,          # 確率出力を使うため必須
    random_state=RANDOM_SEED,
)

X_values = X.values
test_values = test.values
y_values = y.values

# =========================
# 6) 5モデル OOFスタッキング（ベースモデル）
# =========================

skf = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_SEED
)

# oof_meta:
# 0:XGB, 1:LGB, 2:CAT, 3:RF, 4:SVM
oof_meta = np.zeros((X_values.shape[0], 5))
test_meta = np.zeros((test_values.shape[0], 5))

for fold, (tr_idx, va_idx) in enumerate(skf.split(X_values, y_values)):
    print(f"\n[Base Fold {fold+1}/{N_SPLITS}]")

    X_tr, y_tr = X_values[tr_idx], y_values[tr_idx]
    X_va, y_va = X_values[va_idx], y_values[va_idx]

    # --- XGBoost ---
    model_xgb = XGBClassifier(**xgb_params)
    model_xgb.fit(X_tr, y_tr)
    va_pred_xgb = model_xgb.predict_proba(X_va)[:, 1]
    te_pred_xgb = model_xgb.predict_proba(test_values)[:, 1]
    oof_meta[va_idx, 0] = va_pred_xgb
    test_meta[:, 0] += te_pred_xgb / N_SPLITS

    # --- LightGBM ---
    model_lgb = LGBMClassifier(**lgb_params)
    model_lgb.fit(X_tr, y_tr)
    va_pred_lgb = model_lgb.predict_proba(X_va)[:, 1]
    te_pred_lgb = model_lgb.predict_proba(test_values)[:, 1]
    oof_meta[va_idx, 1] = va_pred_lgb
    test_meta[:, 1] += te_pred_lgb / N_SPLITS

    # --- CatBoost ---
    model_cat = CatBoostClassifier(**cat_params)
    model_cat.fit(X_tr, y_tr)
    va_pred_cat = model_cat.predict_proba(X_va)[:, 1]
    te_pred_cat = model_cat.predict_proba(test_values)[:, 1]
    oof_meta[va_idx, 2] = va_pred_cat
    test_meta[:, 2] += te_pred_cat / N_SPLITS

    # --- RandomForest ---
    model_rf = RandomForestClassifier(**rf_params)
    model_rf.fit(X_tr, y_tr)
    va_pred_rf = model_rf.predict_proba(X_va)[:, 1]
    te_pred_rf = model_rf.predict_proba(test_values)[:, 1]
    oof_meta[va_idx, 3] = va_pred_rf
    test_meta[:, 3] += te_pred_rf / N_SPLITS

    # --- SVM（StandardScaler + SVC のパイプライン）---
    svm_clf = make_pipeline(
        StandardScaler(),
        SVC(**svm_params)
    )
    svm_clf.fit(X_tr, y_tr)
    va_pred_svm = svm_clf.predict_proba(X_va)[:, 1]
    te_pred_svm = svm_clf.predict_proba(test_values)[:, 1]
    oof_meta[va_idx, 4] = va_pred_svm
    test_meta[:, 4] += te_pred_svm / N_SPLITS

# =========================
# 7) メタモデル: OOFで評価（Logistic Regression）
# =========================

skf_meta = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_SEED
)

meta_oof_proba = np.zeros(X_values.shape[0])

for tr_idx, va_idx in skf_meta.split(oof_meta, y_values):
    X_tr_meta = oof_meta[tr_idx]
    y_tr_meta = y_values[tr_idx]
    X_va_meta = oof_meta[va_idx]

    meta = LogisticRegression(
        max_iter=1000,
        C=0.5,              # 正則化やや強め
        solver='lbfgs'
    )
    meta.fit(X_tr_meta, y_tr_meta)
    meta_oof_proba[va_idx] = meta.predict_proba(X_va_meta)[:, 1]

# PRカーブから最適閾値（F1最大）
precision, recall, thresholds = precision_recall_curve(y_values, meta_oof_proba)
f1_scores = 2 * precision * recall / (precision + recall + 1e-9)
best_idx = np.nanargmax(f1_scores)
best_thr = thresholds[best_idx] if best_idx < len(thresholds) else 0.5

meta_oof_pred = (meta_oof_proba >= best_thr).astype(int)
meta_oof_f1 = f1_score(y_values, meta_oof_pred)

print(f"\n[Meta-LogReg 5models+SVM] CV-based Best threshold: {best_thr:.4f}")
print(f"[Meta-LogReg 5models+SVM] CV-based OOF F1: {meta_oof_f1:.5f}")

# =========================
# 8) 全データでメタモデル再学習 → テスト予測
# =========================

final_meta = LogisticRegression(
    max_iter=1000,
    C=0.5,
    solver='lbfgs'
)
final_meta.fit(oof_meta, y_values)

test_proba = final_meta.predict_proba(test_meta)[:, 1]
test_pred = (test_proba >= best_thr).astype(int)

# =========================
# 9) 提出ファイル作成
# =========================
sub = pd.DataFrame({
    "企業ID": test.index,
    "購入フラグ": test_pred
})
sub.to_csv("submission_stacking_5models_svm.csv", index=False, header=False)
print("\nSaved: submission_stacking_5models_svm.csv")
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------

