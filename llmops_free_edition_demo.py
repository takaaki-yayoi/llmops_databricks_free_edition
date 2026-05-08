# Databricks notebook source
# MAGIC %md
# MAGIC # Databricks Free EditionでLLMOpsを体験する
# MAGIC
# MAGIC `samples.bakehouse.media_customer_reviews` のレビューから構造化インサイトを抽出するエージェントを題材に、
# MAGIC LLMOpsのコアループ (開発 → プロンプト管理 → 評価 → デプロイ → 運用) を Free Edition で1周します。

# COMMAND ----------
# MAGIC %md
# MAGIC ## 0. セットアップ

# COMMAND ----------
# MAGIC %pip install -qU mlflow "databricks-sdk[openai]" databricks-agents
# MAGIC dbutils.library.restartPython()

# COMMAND ----------
import json
import os
import mlflow
import pandas as pd
from databricks.sdk import WorkspaceClient

# Free Edition のレート制限対策: 評価時の並列度を下げる
# (デフォルトの並列実行だと推論呼び出しがレート制限で失敗しやすい)
os.environ["MLFLOW_GENAI_EVAL_MAX_WORKERS"] = "1"
os.environ["MLFLOW_GENAI_EVAL_MAX_SCORER_WORKERS"] = "1"

# 設定 (環境に合わせて変更してください)
CATALOG = "workspace"
SCHEMA = "llmops_demo"
MODEL_ENDPOINT = "databricks-meta-llama-3-3-70b-instruct"
PROMPT_NAME = f"{CATALOG}.{SCHEMA}.review_insight_extractor"
INSIGHTS_TABLE = f"{CATALOG}.{SCHEMA}.review_insights"

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
mlflow.set_registry_uri("databricks-uc")

w = WorkspaceClient()
client = w.serving_endpoints.get_open_ai_client()

# エクスペリメントはノートブックに自動で紐づくため明示設定はしない
# (右サイドバーの Experiments パネルからトレースを直接確認できる)

print(f"Model: {MODEL_ENDPOINT}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 1. 題材データの確認
# MAGIC
# MAGIC `samples.bakehouse.media_customer_reviews` には架空のベーカリーチェーンに対する顧客レビューが入っています。

# COMMAND ----------
display(
    spark.table("samples.bakehouse.media_customer_reviews").limit(5)
)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 1: ベースライン開発 + MLflow Tracing
# MAGIC
# MAGIC まずは Foundation Model APIs を直接叩いて、レビューから構造化インサイトを抽出する関数を書きます。
# MAGIC `@mlflow.trace` を最初から付けることで、後段の評価・運用で必要なトレースが自動的に蓄積されます。

# COMMAND ----------
PROMPT_V1 = """あなたはベーカリーチェーンの顧客レビュー分析アシスタントです。
以下のレビューから感情、トピック、改善示唆を抽出してください。

出力は以下のJSONスキーマに厳密に従ってください:
{
  "sentiment": "positive | negative | neutral のいずれか",
  "topics": ["product_quality", "service", "price", "atmosphere" のいずれか1つ以上"],
  "actionable_feedback": {
    "has_feedback": true または false,
    "summary": "改善示唆の短い要約。なければ空文字"
  }
}

レビュー:
{{review}}
"""

@mlflow.trace
def extract_insights(review: str) -> dict:
    # MLflow Prompt Registry 規約に合わせた {{review}} を置換
    prompt = PROMPT_V1.replace("{{review}}", review)
    response = client.chat.completions.create(
        model=MODEL_ENDPOINT,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)

# 動作確認
sample = (
    spark.table("samples.bakehouse.media_customer_reviews")
    .select("review")
    .limit(1)
    .collect()[0]["review"]
)
print("レビュー:", sample[:200])
print()
print("抽出結果:")
print(json.dumps(extract_insights(sample), indent=2, ensure_ascii=False))

# COMMAND ----------
# MAGIC %md
# MAGIC 左サイドバーの「Experiments」からトレースを確認できます。
# MAGIC 入力プロンプト、出力、トークン数、レイテンシが1呼び出しごとに記録されます。

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 2: プロンプトのバージョン管理 (MLflow Prompt Registry)
# MAGIC
# MAGIC プロンプトをコードに直書きせず、Unity Catalogで管理します。エイリアスでバージョンを切り替えられるので、
# MAGIC 評価で v2 が良ければ alias を付け替えるだけでデプロイされます。

# COMMAND ----------
# v1: シンプル版を登録
prompt_v1 = mlflow.genai.register_prompt(
    name=PROMPT_NAME,
    template=PROMPT_V1,
    commit_message="v1: ベースライン (指示のみ)",
)
print(f"v1 registered: version={prompt_v1.version}")

# v2: few-shot を追加した版を登録
# 注意: few-shot 例は {{review}} プレースホルダの「前」に配置する
# (後ろに置くとLLMが本物のレビューと例の続きを区別できなくなる)
FEW_SHOT_EXAMPLES = """
# 例1
レビュー: ケーキは美味しかったけど店員の対応が冷たかった
出力: {"sentiment": "neutral", "topics": ["product_quality", "service"], "actionable_feedback": {"has_feedback": true, "summary": "接客態度の改善が必要"}}

# 例2
レビュー: パンが絶品!また来ます
出力: {"sentiment": "positive", "topics": ["product_quality"], "actionable_feedback": {"has_feedback": false, "summary": ""}}

# 評価対象
"""

PROMPT_V2 = PROMPT_V1.replace("レビュー:\n{{review}}", FEW_SHOT_EXAMPLES + "レビュー:\n{{review}}")

prompt_v2 = mlflow.genai.register_prompt(
    name=PROMPT_NAME,
    template=PROMPT_V2,
    commit_message="v2: few-shot 2件を追加",
)
print(f"v2 registered: version={prompt_v2.version}")

# alias でバージョンを管理
mlflow.genai.set_prompt_alias(
    name=PROMPT_NAME, alias="production", version=prompt_v1.version
)
mlflow.genai.set_prompt_alias(
    name=PROMPT_NAME, alias="staging", version=prompt_v2.version
)
print(f"alias 'production' -> v{prompt_v1.version}, 'staging' -> v{prompt_v2.version}")

# COMMAND ----------
# alias 経由で取得して使う関数に書き換え
def extract_with_alias(review: str, alias: str = "production") -> dict:
    prompt_obj = mlflow.genai.load_prompt(f"prompts:/{PROMPT_NAME}@{alias}")
    prompt = prompt_obj.format(review=review)
    response = client.chat.completions.create(
        model=MODEL_ENDPOINT,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 3: 評価ループ (ゴールデンデータ + make_judge)
# MAGIC
# MAGIC 15件のゴールデンデータでv1とv2を比較します。`make_judge` で3つの観点でジャッジを作成:
# MAGIC
# MAGIC 1. **sentiment_correctness**: 感情ラベルが期待値と一致するか
# MAGIC 2. **schema_compliance**: JSONスキーマに準拠しているか
# MAGIC 3. **topics_relevance**: トピックが妥当か

# COMMAND ----------
golden_data = pd.DataFrame([
    {"review": "ここのチョコレートクロワッサンは絶品!毎週通っています。",
     "expected_sentiment": "positive", "expected_topics": ["product_quality"]},
    {"review": "値段が高すぎる。味は普通なのに...",
     "expected_sentiment": "negative", "expected_topics": ["price", "product_quality"]},
    {"review": "店員さんがとても親切で、おすすめを丁寧に教えてくれました。",
     "expected_sentiment": "positive", "expected_topics": ["service"]},
    {"review": "店内が狭くて子連れだと厳しい。味は悪くないのに残念。",
     "expected_sentiment": "neutral", "expected_topics": ["atmosphere", "product_quality"]},
    {"review": "並んだのにパンが売り切れてた。事前案内がほしい。",
     "expected_sentiment": "negative", "expected_topics": ["service"]},
    {"review": "新作のメロンパン、期待以上の出来でした。",
     "expected_sentiment": "positive", "expected_topics": ["product_quality"]},
    {"review": "可もなく不可もなく。普通のパン屋さん。",
     "expected_sentiment": "neutral", "expected_topics": ["product_quality"]},
    # 以下、複合感情・文脈依存の微妙なケース (few-shot の効きが現れやすい)
    {"review": "美味しいんだけど、待ち時間が長すぎて二度と来たくない",
     "expected_sentiment": "negative", "expected_topics": ["service", "product_quality"]},
    {"review": "値段相応の味かな、特別感はないけど",
     "expected_sentiment": "neutral", "expected_topics": ["product_quality", "price"]},
    {"review": "新作は微妙だったけど、定番のクロワッサンは安定の美味しさ",
     "expected_sentiment": "neutral", "expected_topics": ["product_quality"]},
    {"review": "値段は高いけど、それに見合う品質だと思う",
     "expected_sentiment": "positive", "expected_topics": ["price", "product_quality"]},
    {"review": "対応は丁寧だったけど、肝心のパンが期待外れ",
     "expected_sentiment": "negative", "expected_topics": ["service", "product_quality"]},
])
# 注: 本番運用ではゴールデンデータは30件以上を推奨。本記事では概念実演のため12件にしています。

eval_dataset = [
    {
        "inputs": {"review": row["review"]},
        "expectations": {
            "sentiment": row["expected_sentiment"],
            "topics": row["expected_topics"],
        },
    }
    for _, row in golden_data.iterrows()
]
print(f"ゴールデンデータ: {len(eval_dataset)}件")

# COMMAND ----------
from mlflow.genai.judges import make_judge
from typing import Literal

JUDGE_MODEL = f"databricks:/{MODEL_ENDPOINT}"

sentiment_judge = make_judge(
    name="sentiment_correctness",
    instructions=(
        "あなたは評価者です。{{ inputs }} のレビューに対するエージェントの {{ outputs }} と、"
        "{{ expectations }} に含まれる expected_sentiment を比較してください。"
        "outputs の sentiment フィールドが expected_sentiment と一致していれば 'pass'、そうでなければ 'fail' を返してください。"
    ),
    model=JUDGE_MODEL,
    feedback_value_type=Literal["pass", "fail"],
)

schema_judge = make_judge(
    name="schema_compliance",
    instructions=(
        "{{ outputs }} が以下のJSONスキーマに準拠しているか評価してください: "
        "sentiment が positive/negative/neutral のいずれか、"
        "topics が文字列のリスト、"
        "actionable_feedback が has_feedback (bool) と summary (string) を含むオブジェクト。"
        "全て満たせば 'pass'、欠ければ 'fail'。"
    ),
    model=JUDGE_MODEL,
    feedback_value_type=Literal["pass", "fail"],
)

topics_judge = make_judge(
    name="topics_relevance",
    instructions=(
        "{{ inputs }} のレビュー内容に対して、{{ outputs }} の topics が妥当かを評価してください。"
        "{{ expectations }} の expected_topics と完全一致する必要はありません。"
        "ただし明らかに無関係なトピックが含まれていたり、レビュー本文で言及されている観点が"
        "全く拾えていなければ 'fail'、それ以外は 'pass' としてください。"
    ),
    model=JUDGE_MODEL,
    feedback_value_type=Literal["pass", "fail"],
)

# COMMAND ----------
# v1 (production) で評価
def predict_v1(review: str) -> dict:
    return extract_with_alias(review, alias="production")

def predict_v2(review: str) -> dict:
    return extract_with_alias(review, alias="staging")

# evaluate に渡す predict_fn は、それ自身がトレースのルートになるよう @mlflow.trace で wrap する
# (これがないと内側の LLM 呼び出しがトレースルートになり、リクエストにプロンプト全文が入って
#  v1/v2 でリクエストハッシュが異なってしまうため、ラン間比較ビューでペアリングできない)
predict_v1 = mlflow.trace(predict_v1)
predict_v2 = mlflow.trace(predict_v2)

with mlflow.start_run(run_name="eval_v1"):
    result_v1 = mlflow.genai.evaluate(
        data=eval_dataset,
        predict_fn=predict_v1,
        scorers=[sentiment_judge, schema_judge, topics_judge],
    )

# COMMAND ----------
# v2 (staging) で評価
with mlflow.start_run(run_name="eval_v2"):
    result_v2 = mlflow.genai.evaluate(
        data=eval_dataset,
        predict_fn=predict_v2,
        scorers=[sentiment_judge, schema_judge, topics_judge],
    )

# COMMAND ----------
# MAGIC %md
# MAGIC ### v1とv2の比較を可視化

# COMMAND ----------
import plotly.graph_objects as go

def pass_rate(eval_result, judge_name):
    """ジャッジのpass率を返す (列名のバージョン差異に対応)"""
    df = eval_result.result_df
    for col in [judge_name, f"{judge_name}/value", f"feedback/{judge_name}"]:
        if col in df.columns:
            values = df[col].dropna()
            if values.empty:
                return 0.0
            if values.dtype == bool:
                return values.mean()
            return (values == "pass").mean()
    print(f"WARN: '{judge_name}' 列が見つかりません。利用可能な列: {df.columns.tolist()}")
    return 0.0

# デバッグ用: 実際の列名を確認したい場合は以下をコメントアウト解除
# print("v1 metrics:", result_v1.metrics)
# print("v1 columns:", result_v1.result_df.columns.tolist())

judges = ["sentiment_correctness", "schema_compliance", "topics_relevance"]
v1_rates = [pass_rate(result_v1, j) for j in judges]
v2_rates = [pass_rate(result_v2, j) for j in judges]

fig = go.Figure(data=[
    go.Bar(name="v1 (production)", x=judges, y=v1_rates, marker_color="#FF8C00"),
    go.Bar(name="v2 (staging, few-shot)", x=judges, y=v2_rates, marker_color="#1E90FF"),
])
fig.update_layout(
    title="プロンプトv1 vs v2: ジャッジ別 pass 率",
    yaxis_title="pass 率",
    yaxis=dict(range=[0, 1.05]),
    barmode="group",
)
fig.show()

# COMMAND ----------
# MAGIC %md
# MAGIC v2の方が良ければ alias を付け替えるだけで「本番のプロンプト」が更新されます。
# MAGIC 下記コードは **意図的にコメントアウト** しています。実行すると production alias が v2 に上書きされ、
# MAGIC 以降の v1 vs v2 比較が「v2 vs v2」になってしまうためです。本番運用で v2 への昇格判断ができたら、
# MAGIC コメントアウトを解除して実行してください。

# COMMAND ----------
# v2をproductionに昇格 (再実行時の事故防止のためコメントアウト)
# mlflow.genai.set_prompt_alias(
#     name=PROMPT_NAME, alias="production", version=prompt_v2.version
# )
# print(f"production alias を v{prompt_v2.version} に昇格")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 4: デプロイ (ai_query によるバッチ推論)
# MAGIC
# MAGIC Free Edition ではカスタム Model Serving が使えないため、SQLから直接 Foundation Models を呼べる
# MAGIC `ai_query()` をデプロイ手段として採用します。バッチ用途と相性が良く、
# MAGIC Lakeflow Job でそのまま定期実行できます。

# COMMAND ----------
# productionプロンプトの本文を取得 (SQL に埋め込むため)
prod_prompt_text = mlflow.genai.load_prompt(f"prompts:/{PROMPT_NAME}@production").template

# プレースホルダ {{review}} を SQL で置換できる形式に整形
# (ai_query は文字列連結で渡すため、{{review}} 部分を分離)
prompt_prefix, prompt_suffix = prod_prompt_text.split("{{review}}")

spark.sql(f"""
CREATE OR REPLACE TABLE {INSIGHTS_TABLE} AS
SELECT
  new_id AS review_id,
  franchiseID,
  review_date,
  review,
  ai_query(
    '{MODEL_ENDPOINT}',
    CONCAT(
      '{prompt_prefix.replace("'", "''")}',
      review,
      '{prompt_suffix.replace("'", "''")}'
    )
  ) AS insight
FROM samples.bakehouse.media_customer_reviews
LIMIT 50
""")

display(spark.table(INSIGHTS_TABLE).limit(10))

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 5: 運用と継続改善
# MAGIC
# MAGIC 1. このノートブックを **Lakeflow Jobs** で日次スケジュール → 新規レビューを継続的に処理
# MAGIC 2. 結果テーブル `review_insights` を **AI/BI ダッシュボード** で可視化 (sentiment 比率、topic 分布)
# MAGIC 3. メトリクス劣化を検知したら Step 3 に戻ってプロンプトを改善 → alias 切替で再デプロイ

# COMMAND ----------
# 運用メトリクス (sentiment 比率) のサンプル集計
# insightはJSON文字列で格納されているため from_json でSTRUCT化してからアクセスする
INSIGHT_SCHEMA = "STRUCT<sentiment:STRING, topics:ARRAY<STRING>, actionable_feedback:STRUCT<has_feedback:BOOLEAN, summary:STRING>>"

display(spark.sql(f"""
WITH parsed AS (
  SELECT from_json(insight, '{INSIGHT_SCHEMA}') AS i
  FROM {INSIGHTS_TABLE}
)
SELECT
  i.sentiment AS sentiment,
  COUNT(*) AS cnt,
  ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 1) AS pct
FROM parsed
GROUP BY i.sentiment
ORDER BY cnt DESC
"""))

# COMMAND ----------
# topic 分布
display(spark.sql(f"""
WITH parsed AS (
  SELECT from_json(insight, '{INSIGHT_SCHEMA}') AS i
  FROM {INSIGHTS_TABLE}
)
SELECT
  topic,
  COUNT(*) AS cnt
FROM parsed
LATERAL VIEW EXPLODE(i.topics) AS topic
GROUP BY topic
ORDER BY cnt DESC
"""))

# COMMAND ----------
# MAGIC %md
# MAGIC ## まとめ
# MAGIC
# MAGIC Databricks Free Edition だけで、LLMOps のコアループを1周しました。
# MAGIC
# MAGIC | ステップ | 使った機能 |
# MAGIC |---|---|
# MAGIC | 開発 | Foundation Model APIs + MLflow Tracing |
# MAGIC | プロンプト管理 | MLflow Prompt Registry (UC) |
# MAGIC | 評価 | mlflow.genai.evaluate + make_judge |
# MAGIC | デプロイ | ai_query バッチ推論 |
# MAGIC | 運用 | Lakeflow Jobs + AI/BI ダッシュボード |
# MAGIC
# MAGIC GPUカスタムサービングは Free Edition では使えませんが、
# MAGIC LLMOpsの考え方とツールチェーンは本番と同じものを学習・実践できます。
