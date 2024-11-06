import torch

def test_pytorch():
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA version:", torch.version.cuda)
        print("Number of GPUs:", torch.cuda.device_count())
        print("GPU Name:", torch.cuda.get_device_name(0))

if __name__ == "__main__":
    try:
        test_pytorch()
    except Exception as e:
        print("An error occurred:", e)