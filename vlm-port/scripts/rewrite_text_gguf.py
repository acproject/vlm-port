#!/usr/bin/env python3
# Rewrite the text GGUF: vision special token strings + CONTROL types + add_bos=false.
# Strings must change length -> full rewrite (in-place patch not possible).
import sys
import time

sys.path.insert(0, '/home/acproject/usb/transplant/llama.cpp/gguf-py')
import numpy as np  # noqa: E402
import gguf  # noqa: E402

SRC = sys.argv[1]
DST = sys.argv[2]

FIX_TOKENS = {
    166107: b'<|vision_start|>',
    166108: b'<|vision_end|>',
    166109: b'<|image_pad|>',
}
TOKEN_TYPE_CONTROL = 3  # llama.cpp LLAMA_TOKEN_TYPE_CONTROL

reader = gguf.GGUFReader(SRC, 'r')
arch = reader.get_field(gguf.Keys.General.ARCHITECTURE).contents()
print('arch:', arch)

writer = gguf.GGUFWriter(DST, arch=arch, endianess=reader.endianess)

for field in reader.fields.values():
    if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith('GGUF.'):
        continue
    if field.name == 'tokenizer.ggml.tokens':
        tokens = [bytes(np.asarray(field.parts[i]).reshape(-1)) for i in field.data]
        for tid, s in FIX_TOKENS.items():
            print(f'token {tid}: {tokens[tid]!r} -> {s!r}')
            tokens[tid] = s
        writer.add_key_value(field.name, tokens, gguf.GGUFValueType.ARRAY,
                             sub_type=gguf.GGUFValueType.STRING)
        continue
    if field.name == 'tokenizer.ggml.token_type':
        types = [int(np.asarray(field.parts[i]).reshape(-1)[0]) for i in field.data]
        for tid in FIX_TOKENS:
            print(f'token_type {tid}: {types[tid]} -> {TOKEN_TYPE_CONTROL}')
            types[tid] = TOKEN_TYPE_CONTROL
        writer.add_key_value(field.name, types, gguf.GGUFValueType.ARRAY,
                             sub_type=gguf.GGUFValueType.INT32)
        continue
    if field.name == 'tokenizer.ggml.add_bos_token':
        print('add_bos_token: True -> False')
        writer.add_key_value(field.name, False, gguf.GGUFValueType.BOOL)
        continue
    val_type = field.types[0]
    sub_type = field.types[-1] if val_type == gguf.GGUFValueType.ARRAY else None
    writer.add_key_value(field.name, field.contents(), val_type, sub_type=sub_type)

for tensor in reader.tensors:
    writer.add_tensor_info(tensor.name, tensor.data.shape, tensor.data.dtype,
                           tensor.data.nbytes, tensor.tensor_type)

writer.write_header_to_file()
writer.write_kv_data_to_file()
writer.write_ti_data_to_file()
t0 = time.time()
for i, tensor in enumerate(reader.tensors):
    writer.write_tensor_data(tensor.data, tensor_endianess=reader.endianess)
    if i % 100 == 0:
        print(f'  tensor {i}/{len(reader.tensors)} ({time.time()-t0:.0f}s)', flush=True)
writer.close()
print(f'wrote {DST} in {time.time()-t0:.0f}s')
