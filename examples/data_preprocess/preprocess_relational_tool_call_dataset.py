"""
Preprocess the relational tool call dataset to parquet format
run :
python -m examples.data_preprocess.preprocess_relational_tool_call_dataset
"""

import argparse
import os
import re
import json
import copy as cp
from tqdm import tqdm
from datasets import Dataset

from verl.utils.hdfs_io import copy, makedirs
from pydantic import BaseModel

EXAMPLES_TRAIN = [  
    ("What is the user name for ID 35?", "Charlie"),
    ("What is the location ID for Miami?", "5"),
    ("What city has location ID 4?", "Houston"),
    ("What is the current time in Miami?", "2023-11-14 1:20 PM"),
    ("What is the current weather in Houston?", "Rainy, Temperature: 55°F"),
    ("what is frank the cat's user id?", "43"),
    ("what is donna's email address?", "donna@example.com"),
    ("what is bob's favorite color?", "orange"),
    ("Name the favorite foods of Alice.", "Pizza, Chocolate, Sushi"),
    ("list the allergens in burger", "gluten, dairy"),
    ("How many calories are in Ice Cream?", "200 calories"),
    ("What is the name of food with id 3?", "Sushi"),
    ("Does the current user live in Chicago?", "yes"),
    ("Where does Eve live? (city name)", "Miami"),
    ("Do Bob and Charlie live in the same city?", "no"),
    ("What is the current user's email address?", "charlie@yahoo.com"),
    ("List the current weather where Alice lives.", "Partly Cloudy, Temperature: 68°F"),
    ("How many users have the favorite color yellow?", "2"),
    ("Find users named 'Donna'. How many are there?", "1"),
    ("What is the city for location ID 5?", "Miami"),
    ("What is Frank The Cat's favorite color?", "yellow"),
    ("Which allergens are in Sushi?", "fish, soy"),
    ("Is it raining in Houston right now?", "yes"),
]

EXAMPLES_TEST = [
    ("What is the city for location ID 1?", "New York"),
    ("What is the name of food with id 6?", "Pasta"),
    ("what is eve's user id?", "42"),
    ("get the current user id", "35"),
    ("How many users by the name of bob?", "1"),
    ("what is alice's email address?", "alice@gmail.com"),
    ("find donna's favorite color", "green"),
    ("weather in LA right now?", "Sunny, Temperature: 75°F"),
    ("time in chicago", "2023-11-14 11:15 AM"),
    ("list the allergens in chocolate", "milk, soy"),
    ("If i eat a serving of pizza, how many calories will I consume?", "285 calories"),
    ("what is the current users favorite color?", "yellow"),
    ("eve ate a serving of sushi, what allergens was she exposed to?", "fish, soy"),
    ("Frank who is Even's friend is allergic to dairy. Can he eat the salad?", "yes"),
    ("what is the current users favorite color and name?", "yellow and Charlie"),
    ("whats the name of the city where bob lives?", "Los Angeles"),
    ("Donna is about to go outside. Does she need an umbrella?", "yes"),
    ("Is it likely that Donna is awake right now?", "yes"),
    ("do alice and charlie use the same email provider?", "no"),
    ("Is it likely that Donna is outside with an umbrella at this time?", "yes"),
    ("do bob and alice live in the same city?", "no"),
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="~/data/relational_tool_call")
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    train_dataset_list, test_dataset_list = [], []
    data_source = "relational_tool_call"
    agent_name = "relational_tool_call_agent"
    system_prompt = "Please answer the user's question by using the tools provided. Do not guess the answer. Keep in mind that entities like users,foods and locations have both a name and an ID, which are not the same."

    FUNCTION_NAMES = [
        "get_user_name",
        "list_user_ids",
        "find_users_by_name",
        "find_locations_by_name",
        "find_foods_by_name",
        "get_user_email",
        "get_user_location",
        "get_user_favorite_color",
        "get_user_favorite_foods",
        "get_weather_at_location",
        "get_city_for_location",
        "get_current_time_for_location",
        "get_current_weather_for_location",
        "get_food_name",
        "get_food_calories",
        "get_food_allergic_ingredients",
        "get_current_user_id",
    ]

    for split, tasks in [("train", EXAMPLES_TRAIN), ("test", EXAMPLES_TEST)]:
        for idx, task in tqdm(enumerate(tasks), total=len(tasks), desc=f"Processing `{split}` dataset"):
            
            question, answer = task

            data = {
                "data_source": data_source,
                "agent_name": agent_name,
                "prompt": [
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": question,
                    },
                ],
                "ability": "relational_tool_call",
                "reward_model": {"style": "rule", "ground_truth": answer},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": answer,
                    "question": question,
                    "need_tools_kwargs": True,
                    "tools_kwargs": {
                        **{f"{fn_name}": {
                            "create_kwargs": {"ground_truth": answer},
                        } for fn_name in FUNCTION_NAMES},
                    },
                }
            }

            if split == "train":
                train_dataset_list.append(data)
            else:
                test_dataset_list.append(data)
            # import pdb; pdb.set_trace()

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset = Dataset.from_list(train_dataset_list)
    test_dataset = Dataset.from_list(test_dataset_list)

    print(train_dataset[0]['extra_info']['question'])
    print(test_dataset[0]['extra_info']['question'])

    print(f"train dataset len : {len(train_dataset)}")
    print(f"test dataset len : {len(test_dataset)}")
    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
        
