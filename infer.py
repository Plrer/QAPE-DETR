"""QAPE-DETR single-image inference; uses numeric class IDs."""
import argparse
from pathlib import Path
import torch
from PIL import Image, ImageDraw
from torchvision.transforms import functional as TF
from engine.core import YAMLConfig


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('-c', '--config', default='configs/qape_detr.yml')
    parser.add_argument('-r', '--checkpoint', required=True)
    parser.add_argument('-i', '--image', required=True)
    parser.add_argument('-o', '--output', default='outputs/prediction.jpg')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threshold', type=float, default=0.45)
    args = parser.parse_args()
    cfg = YAMLConfig(args.config)
    cfg.yaml_cfg['HGNetv2']['pretrained'] = False
    model = cfg.model
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    state = checkpoint['ema']['module'] if checkpoint.get('ema') is not None else checkpoint.get('model', checkpoint)
    model.load_state_dict(state, strict=True)
    model = model.to(args.device).eval()
    postprocessor = cfg.postprocessor.to(args.device).eval()
    image = Image.open(args.image).convert('RGB')
    size = cfg.yaml_cfg.get('eval_spatial_size', [640, 640])
    tensor = TF.to_tensor(TF.resize(image, size)).unsqueeze(0).to(args.device)
    with torch.inference_mode():
        prediction = postprocessor(model(tensor), torch.tensor([image.size], device=args.device))[0]
    draw = ImageDraw.Draw(image)
    for label, box, score in zip(prediction['labels'], prediction['boxes'], prediction['scores']):
        if score.item() < args.threshold:
            continue
        coords = box.cpu().tolist()
        draw.rectangle(coords, outline='red', width=2)
        draw.text(coords[:2], f'{label.item()} {score.item():.3f}', fill='red')
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output)
    print(f'Saved: {args.output}')


if __name__ == '__main__':
    main()
