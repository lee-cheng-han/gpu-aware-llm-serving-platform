# Physical Paged KV and Reference Attention

`TorchPagedKVCache` allocates contiguous key and value tensors with layout
`[layer, block, KV head, token-in-block, head dimension]`. Each sequence receives a bounded
`KVBlockTable`; append operations copy complete model-layer K/V slices across page boundaries
and reject writes beyond reserved capacity before modifying storage.

The reference attention path gathers different-length block tables into one padded batch,
builds an explicit validity mask, and invokes PyTorch scaled-dot-product attention. This
proves page layout, block-table translation, masking, and cleanup semantics. It deliberately
copies during gather and currently requires equal query/KV head counts.

CPU tensor tests run whenever PyTorch is installed. On a CUDA host, run:

```bash
make test-paged-kv-gpu
```

Set `PAGED_KV_VALIDATION_OUTPUT` to write a machine-readable device and allocation report.
The default Hugging Face token backend still uses `DynamicCache`; model-level parity and a
direct page-consuming kernel remain required before the project calls the serving path
end-to-end PagedAttention.
