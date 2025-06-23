import torch
import torch.nn as nn
from detr_nested_tensor import NestedTensor, nested_tensor_from_tensor_list
from config import config
from detr_model import detr

class DETRWithCaption(nn.Module):
    def __init__(
        self,
        detr,
        vocab_size=config.VOCAB_SIZE,
        max_len=config.CONTEXT_LEN,
        freeze_detr=False,
        num_encoder_layers=1,
        num_decoder_layers=6,
        nhead=None
    ):
        super().__init__()
        self.detr = detr

        # feature dimension and heads
        d_model = detr.transformer.d_model
        nhead = nhead or detr.transformer.nhead

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead), num_encoder_layers
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model, nhead), num_decoder_layers
        )

        # token embedding + positional
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb   = nn.Parameter(torch.randn(max_len, d_model))

        self.out_proj = nn.Linear(d_model, vocab_size)

        self.max_len = max_len

        self.bos_id = 49406
        self.eos_id = 49407
        self.pad_id = 49408

    def _generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
        return mask.to(self.pos_emb.device)

    def forward(self, samples: NestedTensor, tgt_tokens=None):
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)

        # DETR forward
        features, pos = self.detr.backbone(samples)
        src, mask = features[-1].decompose()
        src_proj = self.detr.input_proj(src)
        hs, _ = self.detr.transformer(
            src_proj, mask,
            self.detr.query_embed.weight,
            pos[-1]
        )
        # hs[-1]: [batch, num_queries, d_model]
        visual_seq = hs[-1].permute(1, 0, 2)  # [S, B, D]

        query_pos = self.detr.query_embed.weight
        enc_in = visual_seq + query_pos.unsqueeze(1)      # [S, B, D]
        memory = self.encoder(enc_in, src_key_padding_mask=None)

        # if no tgt, do (greedy) generation
        if tgt_tokens is None:
            return self.generate_caption(memory, None)

        # preparing tgt embeddings
        B, T = tgt_tokens.shape
        tgt = self.token_emb(tgt_tokens)              # [B, T, D]
        
        tgt = tgt + self.pos_emb[:T, :].unsqueeze(0)  # [B, T, D]
        tgt = tgt.permute(1, 0, 2)                    # [T, B, D]

        # masks
        tgt_mask = self._generate_square_subsequent_mask(T)
        tgt_kpm = (tgt_tokens == self.pad_id)

        # full transformer forward
        dec_out = self.decoder(
            tgt, memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_kpm,
            memory_key_padding_mask=None
        )  # [T, B, D]

        logits = self.out_proj(dec_out)            # [T, B, V]
        return logits.permute(1,0,2)               # [B, T, V]


    @torch.no_grad()
    def generate_caption(self, memory, memory_mask):
        B = memory.size(1)
        ys = torch.full((B, 1), self.bos_id, dtype=torch.long, device=memory.device)
        finished = torch.zeros(B, dtype=torch.bool, device=memory.device)
    
        for i in range(self.max_len - 1):
            # building tgt
            tgt = self.token_emb(ys) + self.pos_emb[:ys.size(1)].unsqueeze(0)  # [B, cur_len, D]
            tgt = tgt.permute(1, 0, 2)                                         # [T, B, D]
    
            # masks
            tgt_mask = self._generate_square_subsequent_mask(tgt.size(0))
            tgt_kpm  = (ys == self.pad_id)  
    
            dec_out = self.decoder(
                tgt, memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_kpm,
                memory_key_padding_mask=memory_mask
            )  # [T, B, D]
    
            next_logits = self.out_proj(dec_out[-1])  # [B, V]
            next_word   = next_logits.argmax(-1, keepdim=True)  # [B,1]
            ys = torch.cat([ys, next_word], dim=1)
    
            finished |= (next_word.squeeze(-1) == self.eos_id)
            if finished.all():
                break
    
        return ys  # [B, <=max_len]



model = DETRWithCaption(detr)
model = model


def load_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Checkpoint loaded. Resuming training from epoch 12")

load_checkpoint(model, 'src\DETR_CAP012.pth')