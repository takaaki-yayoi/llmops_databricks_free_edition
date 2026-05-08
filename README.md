# 概要

Databricks Free Edition だけで LLMOps のコアループ (開発 → プロンプト管理 → 評価 → デプロイ → 運用) を1周通すサンプルノートブックです。`samples.bakehouse.media_customer_reviews` のレビューデータを題材に、感情・トピック・改善示唆を JSON で抽出するエージェントを題材として扱います。

GPU 専用の Model Serving エンドポイントなど一部の機能は Free Edition では使えませんが、LLMOps の考え方とツールチェーンは本番環境とほぼ同じものを学習・実践できます。

# 関連記事

- Qiita: [Databricks Free Edition で LLMOps を体験する](https://qiita.com/taka_yayoi) (公開後にURL差し替え)

# ノートブック構成

1セルずつ順に実行する想定です。

| ステップ | 内容 | 使用機能 |
|---|---|---|
| Step 1 | ベースラインのエージェント開発とトレース取得 | Foundation Model APIs + MLflow Tracing |
| Step 2 | プロンプトを Unity Catalog に登録してバージョン管理 | MLflow Prompt Registry |
| Step 3 | ジャッジ定義と v1 / v2 の評価・比較 | make_judge + mlflow.genai.evaluate |
| Step 4 | SQL からのバッチ推論 | ai_query 関数 |
| Step 5 | 集計クエリと運用ループの設計 | Lakeflow Jobs + AI/BI ダッシュボード (構成方針のみ) |

# 前提条件

- Databricks Free Edition または Trial アカウント
- Unity Catalog の任意スキーマに `CREATE TABLE` 権限 (デフォルトでは `workspace.llmops_demo` を使用)
- Foundation Model APIs の `databricks-meta-llama-3-3-70b-instruct` エンドポイントが利用可能であること

# セットアップと実行

1. このリポジトリを Databricks ワークスペースにインポート (Workspace > Repos > Add Repo)
2. `llmops_free_edition_demo.py` をノートブックとして開く
3. 先頭セルの `CATALOG` `SCHEMA` を必要に応じて変更
4. 「すべてを実行」で1セルずつ進む

評価ループはレート制限を考慮して `MLFLOW_GENAI_EVAL_MAX_WORKERS=1` で直列化してあるため、Step 3 の評価は数分かかります。

# 実装上のポイント

書く過程で踏んだ落とし穴を共有しておきます。

**MLflow Prompt Registry の変数構文は `{{review}}` (二重中括弧)**: Python `str.format` の `{review}` (単一中括弧) や、`{{}}` をリテラル中括弧のエスケープとして使う書き方とは別規約です。プロンプト内に JSON サンプルを書く場合はリテラル中括弧 `{` `}` をそのまま書きます。

**`predict_fn` には `@mlflow.trace` を付ける**: これがないとトレースのルートが内側の LLM 呼び出しになり、リクエストにプロンプト全文が記録されてしまいます。v1 と v2 でプロンプトが違うため、リクエストハッシュが一致せず、UI のラン間比較ビューでペアリングが効かなくなります。

**ai_query の `responseFormat` はトップレベル1フィールド制約あり**: 今回のように `sentiment` / `topics` / `actionable_feedback` がフラットに並ぶ JSON だと噛み合わせが難しいので、生 JSON 文字列で受けて後段で `from_json` でパースする方が素直です。

**Free Edition のサーバレスでは `spark.conf.set` 不可**: 設定はノートブック先頭のグローバル変数とローカル変数で管理しています。

# ライセンス

MIT
