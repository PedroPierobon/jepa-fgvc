import os
import subprocess
import time

# --- Configurações ---
CANDIDATE_GPUS = [0, 1]      # GPUs que você quer disputar
MAX_MEM_MB = 1000            # VRAM máxima usada para considerar livre (em MB)
MAX_UTIL_PCT = 10            # Utilização de computação máxima tolerada (%)
CHECK_INTERVAL = 30          # Intervalo entre checagens (segundos)
REQUIRED_CONSECUTIVE = 3     # Ciclos seguidos ociosa para evitar falsos positivos

# Comando encadeado: executa o teste apenas se o treino terminar com sucesso (&&)
COMMAND_TO_RUN = "python finetune_hierarchical.py --data_dir ./UFPR-VeSV --split_fold 0 --backbone convnext_base --lejepa_checkpoint ./checkpoints/convnext_lejepa/lejepa_encoder_fold0.pth --batch_size 32 --lr_backbone 3e-5 --lr_head 5e-4 --epochs 30 --amp --exp_name 'ft_lejepa_convnext_fold0' --output_dir ./checkpoints_ft/convnext_lejepa_ft"


def get_all_gpus_status():
    """Consulta uso de VRAM e computação de todas as GPUs via nvidia-smi."""
    query = "index,memory.used,utilization.gpu"
    cmd = f"nvidia-smi --query-gpu={query} --format=csv,noheader,nounits"
    gpu_data = {}
    try:
        output = subprocess.check_output(cmd.split()).decode("utf-8").strip()
        for line in output.splitlines():
            if not line.strip():
                continue
            idx, mem, util = [int(x.strip()) for x in line.split(",")]
            gpu_data[idx] = {"mem": mem, "util": util}
        return gpu_data
    except Exception as e:
        print(f"[-] Erro ao executar nvidia-smi: {e}")
        return {}


def wait_and_allocate():
    print(f"[*] Monitorando GPUs {CANDIDATE_GPUS}...")
    print(f"[*] Critério de ociosidade: VRAM < {MAX_MEM_MB}MB e Util < {MAX_UTIL_PCT}% por {REQUIRED_CONSECUTIVE} ciclos.")
    
    # Contador de ciclos ociosos por GPU: {0: 0, 1: 0}
    idle_counters = {gpu_id: 0 for gpu_id in CANDIDATE_GPUS}

    while True:
        status = get_all_gpus_status()
        timestamp = time.strftime("%H:%M:%S")

        for gpu_id in CANDIDATE_GPUS:
            if gpu_id not in status:
                continue

            mem = status[gpu_id]["mem"]
            util = status[gpu_id]["util"]

            # Verifica se atende aos limites de ociosidade
            if mem < MAX_MEM_MB and util < MAX_UTIL_PCT:
                idle_counters[gpu_id] += 1
            else:
                idle_counters[gpu_id] = 0

            print(
                f"[{timestamp}] GPU {gpu_id} | VRAM: {mem:>5} MB | "
                f"Util: {util:>3}% | Ciclos livres: {idle_counters[gpu_id]}/{REQUIRED_CONSECUTIVE}"
            )

            # Se alguma GPU atingiu a estabilidade necessária
            if idle_counters[gpu_id] >= REQUIRED_CONSECUTIVE:
                print(f"\n[+] GPU {gpu_id} está livre e estável!")
                print(f"[+] Alocando pipeline na GPU {gpu_id}: `{COMMAND_TO_RUN}`\n")

                # Injeta a GPU selecionada no ambiente de execução
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

                # Executa o pipeline e aguarda a finalização
                process = subprocess.run(COMMAND_TO_RUN, shell=True, env=env)
                print(f"\n[*] Processo finalizado com código de retorno: {process.returncode}")
                return

        print("-" * 55)
        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    wait_and_allocate()
