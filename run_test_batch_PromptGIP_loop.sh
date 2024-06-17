for i in {0..20..1}; do
    echo $i
    CUDA_VISIBLE_DEVICES=0 python test_PromptGIP_customized.py --model mae_vit_large_patch16_input256 --output_dir results/PromptGIP_released --ckpt pretrained_models/PromptGIP-checkpoint.pth --prompt_id $i --degradation_type GaussianNoise
done

for i in {0..20..1}; do
    echo $i
    CUDA_VISIBLE_DEVICES=0 python test_PromptGIP_customized.py --model mae_vit_large_patch16_input256 --output_dir results/PromptGIP_released --ckpt pretrained_models/PromptGIP-checkpoint.pth --prompt_id $i --degradation_type GaussianBlur
done

for i in {0..20..1}; do
    echo $i
    CUDA_VISIBLE_DEVICES=0 python test_PromptGIP_customized.py --model mae_vit_large_patch16_input256 --output_dir results/PromptGIP_released --ckpt pretrained_models/PromptGIP-checkpoint.pth --prompt_id $i --degradation_type JPEG
done

for i in {0..20..1}; do
    echo $i
    CUDA_VISIBLE_DEVICES=0 python test_PromptGIP_customized.py --model mae_vit_large_patch16_input256 --output_dir results/PromptGIP_released --ckpt pretrained_models/PromptGIP-checkpoint.pth --prompt_id $i --degradation_type Rain
done

for i in {0..20..1}; do
    echo $i
    CUDA_VISIBLE_DEVICES=0 python test_PromptGIP_customized.py --model mae_vit_large_patch16_input256 --output_dir results/PromptGIP_released --ckpt pretrained_models/PromptGIP-checkpoint.pth --prompt_id $i --degradation_type SPNoise
done

for i in {0..20..1}; do
    echo $i
    CUDA_VISIBLE_DEVICES=0 python test_PromptGIP_customized.py --model mae_vit_large_patch16_input256 --output_dir results/PromptGIP_released --ckpt pretrained_models/PromptGIP-checkpoint.pth --prompt_id $i --degradation_type LowLight
done

for i in {0..20..1}; do
    echo $i
    CUDA_VISIBLE_DEVICES=0 python test_PromptGIP_customized.py --model mae_vit_large_patch16_input256 --output_dir results/PromptGIP_released --ckpt pretrained_models/PromptGIP-checkpoint.pth --prompt_id $i --degradation_type PoissonNoise
done

for i in {0..20..1}; do
    echo $i
    CUDA_VISIBLE_DEVICES=0 python test_PromptGIP_customized.py --model mae_vit_large_patch16_input256 --output_dir results/PromptGIP_released --ckpt pretrained_models/PromptGIP-checkpoint.pth --prompt_id $i --degradation_type Ringing
done

for i in {0..20..1}; do
    echo $i
    CUDA_VISIBLE_DEVICES=0 python test_PromptGIP_customized.py --model mae_vit_large_patch16_input256 --output_dir results/PromptGIP_released --ckpt pretrained_models/PromptGIP-checkpoint.pth --prompt_id $i --degradation_type r_l
done

for i in {0..20..1}; do
    echo $i
    CUDA_VISIBLE_DEVICES=0 python test_PromptGIP_customized.py --model mae_vit_large_patch16_input256 --output_dir results/PromptGIP_released --ckpt pretrained_models/PromptGIP-checkpoint.pth --prompt_id $i --degradation_type Inpainting
done

for i in {0..20..1}; do
    echo $i
    CUDA_VISIBLE_DEVICES=0 python test_PromptGIP_customized.py --model mae_vit_large_patch16_input256 --output_dir results/PromptGIP_released --ckpt pretrained_models/PromptGIP-checkpoint.pth --prompt_id $i --degradation_type Test100
done

for i in {51..301..10}; do
    echo $i
    CUDA_VISIBLE_DEVICES=0 python test_PromptGIP_customized.py --model mae_vit_large_patch16_input256 --output_dir results/PromptGIP_released --ckpt pretrained_models/PromptGIP-checkpoint.pth --prompt_id $i --degradation_type SOTS
done

for i in {0..20..1}; do
    echo $i
    CUDA_VISIBLE_DEVICES=0 python test_PromptGIP_customized.py --model mae_vit_large_patch16_input256 --output_dir results/PromptGIP_released --ckpt pretrained_models/PromptGIP-checkpoint.pth --prompt_id $i --degradation_type LOL
done

for i in {0..20..1}; do
    echo $i
    CUDA_VISIBLE_DEVICES=0 python test_PromptGIP_customized.py --model mae_vit_large_patch16_input256 --output_dir results/PromptGIP_released --ckpt pretrained_models/PromptGIP-checkpoint.pth --prompt_id $i --degradation_type Canny
done

for i in {0..20..1}; do
    echo $i
    CUDA_VISIBLE_DEVICES=0 python test_PromptGIP_customized.py --model mae_vit_large_patch16_input256 --output_dir results/PromptGIP_released --ckpt pretrained_models/PromptGIP-checkpoint.pth --prompt_id $i --degradation_type LLF
done

for i in {0..20..1}; do
    echo $i
    CUDA_VISIBLE_DEVICES=0 python test_PromptGIP_customized.py --model mae_vit_large_patch16_input256 --output_dir results/PromptGIP_released --ckpt pretrained_models/PromptGIP-checkpoint.pth --prompt_id $i --degradation_type Laplacian
done

for i in {0..20..1}; do
    echo $i
    CUDA_VISIBLE_DEVICES=0 python test_PromptGIP_customized.py --model mae_vit_large_patch16_input256 --output_dir results/PromptGIP_released --ckpt pretrained_models/PromptGIP-checkpoint.pth --prompt_id $i --degradation_type L0_smooth
done