import os
import torch
from diffusers import StableDiffusion3Pipeline
from diffusers import DiffusionPipeline

# Configurações importantes para economia de memória
LOW_VRAM_MODE = True  # Ativar para GPUs com pouca VRAM (< 8GB)
HALF_PRECISION = True  # Usar precisão pela metade (fp16)

# Configurar variáveis de ambiente (ANTES de inicializar o pipeline)
if LOW_VRAM_MODE:
    # Reduzir ainda mais o tamanho dos blocos alocados
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"
    # os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True' # Alternativa para fragmentação
else:
    # Aumenta um pouco para GPUs com mais VRAM
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"


# Inicializar o pipeline com as otimizações
try:
    if HALF_PRECISION:
        pipe = StableDiffusion3Pipeline.from_pretrained(
            "stabilityai/stable-diffusion-3.5-large",
            torch_dtype=torch.float16,
            variant="fp16",
        )
    else:
        pipe = StableDiffusion3Pipeline.from_pretrained(
            "stabilityai/stable-diffusion-3.5-large",
            # Usar float32 se necessário (mais lento, mais memória)
            torch_dtype=torch.float32,
        )

    if LOW_VRAM_MODE:
        # Otimização de atenção (substitui model_cpu_offload em versões recentes)
        pipe.enable_attention_slicing()
        pipe.enable_vae_slicing()
        # Usa xformers para atenção mais eficiente (se instalado)
        pipe.enable_xformers_memory_efficient_attention()
        # pipe.enable_model_cpu_offload() # Use somente se attention_slicing não estiver disponível
    else:
        pipe.enable_vae_slicing()

    # Gerar imagem com configurações de baixa memória
    image = pipe(
        prompt="an Orgy cene with beautiul women, one blonde, one brunette, and one redhead, in a luxurious mansion",
        num_inference_steps=20,
        guidance_scale=3.0,
        height=512,
        width=512,
    ).images[0]

    image.save("imagem_ex1.png")
    print("Imagem gerada com sucesso!")

except torch.cuda.OutOfMemoryError as e:
    print(f"Erro de memória CUDA: {e}")
    print("Tente reduzir o tamanho da imagem, batch size, ou usar uma máquina com mais memória GPU.")
except Exception as e:
    print(f"Outro erro: {e}")

# Limpeza explícita (importante!)
del pipe
torch.cuda.empty_cache()
