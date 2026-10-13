# Corrected model sources and child layers

The curated models use a fixed single attention head. `heads` and `num_heads` are removed from the model and shared-layer constructors and from the training factory specification. Calls must omit these arguments. Tensor dimensions, parameter names and shapes, and operation order remain those of the original `heads=1` implementation; existing single-head state dictionaries remain compatible with strict loading. The singleton head axis is retained internally to preserve the original computations. Multi-head execution is not supported.

The initial curation incorrectly chose `ver_unetG/model_unetT.py` for Stage1. The author's correction supersedes that choice. The old block/k-hop model closure and its separate dependency group were removed; original research checkouts were preserved.

| Stage | Verified source | Curated entry | Levels |
|---|---|---|---|
| 1 | `D:\RDL_TC_MODEL\gnn\ver_unet_scale\model_unet.py` | `agu_james/training/models/coarse/model_unet.py` | 3–6 |
| 2–4 | `D:\RDL_TC_MODEL\gnn\ver_autoregressive\model_unet.py` | `agu_james/training/models/ar/model_unet.py` | 3–7 |

The `HeteroGNN` class comparison finds the finest level, encoder relation/key, pooling/upsampling/decoder level lists and the associated per-level default processor list as the differences. Coarse defaults `[16,8,4,2]` for levels6/5/4/3; AR defaults `[1,16,8,4,2]` for levels7/6/5/4/3. The extra finest level accounts for the extra list element. Encoder/GAT, processor/SG branch, normalization, upsampling and decoder implementations are the same. The previous `ver_unet7` and AR `HeteroGNN` classes also have identical ASTs; Stages2–4 now select one AR model class directly.

The five required child sources were compared with parsed ASTs. All are identical between `ver_unet_scale` and `ver_autoregressive`, although two original byte hashes differ due to comments/formatting. They are kept once under `agu_james/training/models/shared/`:

- `gatv2convNG.py`: encoder, pooling and multiscale decoding.
- `gat_edge_before_ffnG_fix.py`: processor GAT/edge update and FFN.
- `gat_edge_ffn_upG_fix.py`: upsampling.
- `processor__blockG.py`: processor branch construction and GAT/SG fusion.
- `SG_module.py`: SG linear-attention branch.

`engine.make_model` switches between the two model roots by stage; it no longer passes `k_hop` or `trans_layer`. Stage1 graph construction still uses mesh3–6, and subsequent stages use mesh3–7. Both corrected source training entries exclude bias and normalization parameters from AdamW decay.

Architecture and training controls are distinct. Existing trainer configuration values remain explicit: Stage1 processor depths `[16,12,8,4]`, FFN ratio1; later stages `[16,16,12,8,6]`, FFN ratio2. These are local training settings in the common factory, not a different child-layer implementation. `training.processor_layers` and `training.ffn_ratio` can override them. Specify matching values when comparing level-only variants or transferring all shape-compatible tensors; full historical checkpoint transfer/retraining has not been executed.

Validation: synthetic coarse and AR forward/backward computations have finite outputs/gradients; Stage2 and Stage3 select the same class; original model-class computations and curated predictions are **exactly equal** on controlled inputs with identical weights for Stages1/2/3. The child-source AST identity is separately asserted. The retained model closure imports in the current PyG environment without direct torch-scatter/torch-sparse dependencies. No full-size atmospheric training or historical checkpoint evaluation is implied.
