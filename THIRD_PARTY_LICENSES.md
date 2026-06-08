# Third-Party Licenses and Attributions

This repository is a modified fork of [dunnolab/laom](https://github.com/dunnolab/laom),
which is licensed under the Apache License 2.0. The project as a whole is
distributed under the Apache License 2.0 (see [`LICENSE`](LICENSE)) and includes
code from the third-party sources listed below, each of which retains its
original license.

## 1. laom (upstream project) — Apache-2.0

- Source: https://github.com/dunnolab/laom
- License: Apache License 2.0 (full text in [`LICENSE`](LICENSE))
- This repository modifies the original work (adds an invertible i-ResNet
  decoder, supporting utilities, and related experiments).

## 2. invertible-resnet — MIT

- File(s): `src/spectral_norm_fc.py`
- Source: https://github.com/jhjacobsen/invertible-resnet
- License: MIT

```
MIT License

Copyright (c) 2019 Jörn Jacobsen

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## 3. Distracting Control Suite (google-research) — Apache-2.0

- File(s): `src/dcs/*`
- Source: https://github.com/google-research/google-research/tree/master/distracting_control
- License: Apache License 2.0 (same text as [`LICENSE`](LICENSE))
- Copyright 2024 The Google Research Authors. Original per-file Apache headers
  are retained in each file.

## Other adapted sources

As noted in the README, data-collection scripts adapt code from
[CleanRL](https://github.com/vwxyzjn/cleanrl) (MIT) and use
[stable-baselines3](https://github.com/DLR-RM/stable-baselines3) (MIT). If any
of that code is copied into this repository (e.g. under
`scripts/data_collection/`), its MIT copyright and permission notice should be
retained in the corresponding files.
