import torch

m = torch.load("/home/daodao/Develop/DartKVMQ/configs/skvq_calib/"
               "qwen3-4b_wikitext2_rod_g128.pt")
ri, cs = m["reorder_indices"], m["cluster_st_inds"]
print("layers:", len(ri))
for l in [0, 16, 31]:
    kb = cs[l][0].tolist()
    sizes = [kb[i + 1] - kb[i] for i in range(len(kb) - 1)]
    print(f"layer {l}: K cluster sizes {sizes}")
    assert sum(sizes) == 1024
idx = ri[0][0]
assert sorted(idx.tolist()) == list(range(1024))
print("reorder file OK: valid permutations, variable-size clusters")
