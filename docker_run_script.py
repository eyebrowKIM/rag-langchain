import subprocess

model_list = [
    "yanolja/EEVE-Korean-Instruct-10.8B-v1.0",
    "saltlux/Ko-Llama3-Luxia-8B",
    "cpm-ai/gemma-ko-v01",
    "beomi/Llama-3-Open-Ko-8B",
    "beomi/gemma-ko-2b",
    "daekeun-ml/phi-2-ko-v0.1",
    "maum-ai/Llama-3-MAAL-8B-Instruct-v0.1",
]

id_list = [3000, 3002]


def run_docker_container(device="1,2", id=3002, model_name="yanolja/EEVE-Korean-Instruct-10.8B-v1.0"):
    device = "device=" + device

    command = """
    docker run -it --restart always --name jskim --gpus '"{}"' -v /data/bentoml/models:/root/bentoml/models -p {}:3000 -it ghcr.io/bentoml/openllm start {} --backend vllm --max-new-tokens 1024
    """.format(
        device, id, model_name
    )

    try:
        # subprocess.run을 사용하여 Docker 커맨드 실행
        result = subprocess.run(command, shell=True, check=True, text=True)
        print("Docker container started successfully.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while trying to start the Docker container: {e}")


if __name__ == "__main__":
    run_docker_container()
