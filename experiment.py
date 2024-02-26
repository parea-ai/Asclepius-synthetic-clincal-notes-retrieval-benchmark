import os
import uuid

import numpy as np
import asyncpg
import psycopg2
from dotenv import load_dotenv
from parea import Parea, trace
from tqdm import tqdm

from evals import gpt_35_turbo_0125_0_shot, gpt_35_turbo_0125_1_shot_false_sample_1, \
    gpt_35_turbo_0125_1_shot_false_sample_2, gpt_35_turbo_0125_2_shot_false_1_false_2, \
    gpt_35_turbo_0125_2_shot_false_2_false_1, hit_rate_top_20, mrr_top_20

load_dotenv()

DB_URL = os.getenv('LANTERN_DB_URL')
TABLE_NAME = 'synthetic'

p = Parea(api_key=os.getenv("PAREA_API_KEY"), project_name='Asclepius-retrieval-benchmark')


@trace(eval_funcs=[
    hit_rate_top_20,
    mrr_top_20,
    # gpt_35_turbo_0125_0_shot,
    # gpt_35_turbo_0125_1_shot_false_sample_1, gpt_35_turbo_0125_1_shot_false_sample_2,
    # gpt_35_turbo_0125_2_shot_false_1_false_2, gpt_35_turbo_0125_2_shot_false_2_false_1,
])
async def get_answer(row_id: int, question: str, emb_model: str, task: str, limit: int = 10) -> list[dict[str, int | str]]:
    aconn = await asyncpg.connect(DB_URL)
    query_vector = f"(SELECT question_embedding_{emb_model} FROM {TABLE_NAME} WHERE id = {row_id})"
    records = await aconn.fetch(
        f"SELECT id, answer "
        f"FROM {TABLE_NAME} "
        f"WHERE task LIKE '{task}'"
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


def load_data(emb_model: str, limit: int, task) -> list[dict[str, int | str]]:
    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()
    cur.execute(f"SELECT id, question FROM {TABLE_NAME} WHERE task LIKE '{task}';")
    records = cur.fetchall()

    cur.close()
    conn.close()
    return [
        {
            "row_id": record[0],
            "question": record[1],
            "emb_model": emb_model,
            "limit": limit,
            "task": task
        }
        for record in records
    ]


if __name__ == "__main__":
    num_data = 400

    tasks = ['Paraphrasing', 'Question Answering']
    embedding_models = ['openai_small_min', 'openai_small_max', 'openai_large_min', 'openai_large_max']

    configs = []
    for emb_model in embedding_models:
        for task in tasks:
            configs.append({"emb_model": emb_model, "task": task})

    pbar = tqdm(configs)
    for config in pbar:
        emb_model = config["emb_model"]
        task = config["task"]
        data = load_data(emb_model, 20, task)
        if num_data:
            np.random.shuffle(data)
            data = data[:num_data]

        experiment_name = f"{emb_model}-{task}".replace(' ', '_')
        if num_data:
            experiment_name += f"-{num_data}-samples"
        experiment_name += f"-{str(uuid.uuid4())[:4]}"

        pbar.set_description(f"Running {experiment_name} ...")

        p.experiment(data=data, func=get_answer).run(name=experiment_name)
