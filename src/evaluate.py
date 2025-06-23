import torch
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

from config import Config
from dataset import clip_coco_dataset, test_loader
from imgcap_model import model

from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.cider.cider import Cider


def clean_and_split(text: str, lower: bool = True):
    if lower:
        text = text.lower()
    tokens = text.split()
    return [t for t in tokens if t not in {"[PAD]", "<|startoftext|>", "<|endoftext|>"}]


def generate_captions():
    cfg = Config()
    device = cfg.DEVICE
    model.to(device).eval()

    refs, hyps = [], []
    smooth_fn = SmoothingFunction().method1

    with torch.no_grad():
        for images, _, ref_ids_batch in tqdm(test_loader, desc="Generating captions"):
            images = images.to(device).float()
            pred_ids_batch = model(images)  ## (B, seq_len)

            for ref_ids_set, pred_ids in zip(ref_ids_batch, pred_ids_batch):
                ref_texts = [
                    clean_and_split(
                        clip_coco_dataset.tokenizer.decode(r, skip_special_tokens=True)
                    )
                    for r in ref_ids_set
                ]
                hyp_text = clean_and_split(
                    clip_coco_dataset.tokenizer.decode(pred_ids.tolist(), skip_special_tokens=True)
                )
                refs.append(ref_texts)
                hyps.append(hyp_text)

    return refs, hyps


def compute_bleu(refs, hyps):
    smooth_fn = SmoothingFunction().method1
    bleu_scores = {}
    for i in range(1, 5):
        weights = tuple((1.0 / i if j < i else 0.0) for j in range(4))
        score = corpus_bleu(refs, hyps, weights=weights, smoothing_function=smooth_fn)
        bleu_scores[f"BLEU-{i}"] = score * 100
    return bleu_scores


def compute_meteor_cider(refs, hyps):
    gts, res = {}, {}
    for idx, (ref_list, hyp_list) in enumerate(zip(refs, hyps)):
        gts[idx] = [" ".join(r) for r in ref_list]
        res[idx] = [" ".join(hyp_list)]

    meteor_scorer = Meteor()
    cider_scorer = Cider()

    meteor_score, _ = meteor_scorer.compute_score(gts, res)
    cider_score, _ = cider_scorer.compute_score(gts, res)

    return meteor_score * 100, cider_score * 100


def main():
    print("Running caption generation...")
    refs, hyps = generate_captions()

    print("\n=== BLEU Scores ===")
    bleu_scores = compute_bleu(refs, hyps)
    for k, v in bleu_scores.items():
        print(f"{k}: {v:.2f}")

    print("\n=== METEOR & CIDEr ===")
    meteor, cider = compute_meteor_cider(refs, hyps)
    print(f"METEOR: {meteor:.2f}")
    print(f"CIDEr: {cider:.2f}")

if __name__ == "__main__":
    main()
