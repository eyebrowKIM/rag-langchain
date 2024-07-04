import os
import json
from dotenv import load_dotenv
from langsmith import Client

load_dotenv()
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_PROJECT'] = 'RAG eval'
os.environ['LANGCHAIN_API_KEY'] = os.getenv("LANGCHAIN_API_KEY")


# JSON 파일을 불러오기 위한 함수
def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

# "Questions" 내의 "question" 값 리스트를 반환하는 함수
def extract_questions_and_answers(json_data):
    question_list = [question['question'] for question in json_data.get('Questions', [])]
    answer_list = [answer["answer"] for answer in json_data.get('Answers', [])]
    return question_list, answer_list

# Replace 'eval/data.json' with the actual path to your JSON file
json_file_path = 'eval/data.json'
data = load_json(json_file_path)
questions, answers = extract_questions_and_answers(data)

inputs = [{"question": q} for q in questions]
outputs = [{"answer": a} for a in answers]


client = Client()
dataset_name = "RAG_eval_test"

dataset = client.create_dataset(
    dataset_name=dataset_name,
    description="QA pairs for RAG evaluation"
)

client.create_examples(
    inputs=inputs,
    outputs=outputs,
    dataset_id=dataset.id
    )