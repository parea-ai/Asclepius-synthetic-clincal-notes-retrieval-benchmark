import os

import numpy as np
from openai import OpenAI
import asyncpg
import psycopg2
from dotenv import load_dotenv
from parea import Parea, trace

from evals import gpt_35_turbo_0125_0_shot, gpt_35_turbo_0125_1_shot_false_sample_1, \
    gpt_35_turbo_0125_1_shot_false_sample_2, gpt_35_turbo_0125_2_shot_false_1_false_2, \
    gpt_35_turbo_0125_2_shot_false_2_false_1, hit_rate_top_20, mrr_top_20

load_dotenv()

DB_URL = os.getenv('LANTERN_DB_URL')
TABLE_NAME = 'synthetic'

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
p = Parea(api_key=os.getenv("PAREA_API_KEY"), project_name='Asclepius-retrieval-benchmark')

TASK = 'Paraphrasing'
# TASK = 'Question Answering'

emb_model = 'jina_base'
# emb_model = 'bge_base'
num_data = None


@trace(eval_funcs=[
    hit_rate_top_20,
    mrr_top_20,
    gpt_35_turbo_0125_0_shot,
    gpt_35_turbo_0125_1_shot_false_sample_1, gpt_35_turbo_0125_1_shot_false_sample_2,
    gpt_35_turbo_0125_2_shot_false_1_false_2, gpt_35_turbo_0125_2_shot_false_2_false_1,
])
async def get_answer(row_id: int, question: str, emb_model: str, limit: int = 10) -> list[dict[str, int | str]]:
    aconn = await asyncpg.connect(DB_URL)
    query_vector = f"(SELECT question_embedding_{emb_model} FROM {TABLE_NAME} WHERE id = {row_id})"
    records = await aconn.fetch(
        f"SELECT id, answer "
        f"FROM {TABLE_NAME} "
        f"WHERE task LIKE '{TASK}'"
        f"ORDER BY cos_dist(answer_embedding_{emb_model}, ARRAY{query_vector}) "
        f"LIMIT {limit}"
    )

    await aconn.close()
    return [
        {
            "id": record[0],
            "answer": record[1]
        }
        for record in records
    ]


def load_data(emb_model: str, limit: int) -> list[dict[str, int | str]]:
    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()
    cur.execute(f"SELECT id, question FROM {TABLE_NAME} WHERE task LIKE '{TASK}';")
    records = cur.fetchall()

    cur.close()
    conn.close()
    return [
        {
            "row_id": record[0],
            "question": record[1],
            "emb_model": emb_model,
            "limit": limit
        }
        for record in records
    ]


data = load_data(emb_model, 20)
if num_data:
    np.random.shuffle(data)
    data = data[:num_data]

p.experiment(
    data=data,
    func=get_answer
)
