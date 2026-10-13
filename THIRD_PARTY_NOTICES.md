# Third-party attribution and license status

The license for author-owned code is **PENDING AUTHOR APPROVAL**. There is deliberately no blanket MIT/Apache license for this mixed-source repository. Publication requires reconciling author-owned and third-party terms; this file does not grant new permissions over either.

## Pangu-Weather and ReadyToGo

Pangu-Weather is attributed to Kaifeng Bi, Lingxi Xie, Hengheng Zhang, Xin Chen, Xiaotao Gu and Qi Tian / Huawei Cloud. The official repository is https://github.com/198808xc/Pangu-Weather; the method citation is Bi et al. (2023), Nature, doi:10.1038/s41586-023-06185-3. Its [license section](https://github.com/198808xc/Pangu-Weather#license) applies CC BY-NC-SA 4.0 to trained parameters; this must not be described as an unrestricted open-source model license.

The supplied local checkout derives from [HaxyMoly/Pangu-Weather-ReadyToGo](https://github.com/HaxyMoly/Pangu-Weather-ReadyToGo). Its README states BY-NC-SA 4.0 and requests citation of Pangu-Weather. The curated inference, validation, subset and map-export utilities retain this upstream attribution. The precise local modifications and original SHA-256 are in `provenance/sources.json`. Preserve the stated attribution/noncommercial/share-alike conditions when reviewing release terms; this file does not infer a different license for unlabelled local additions. Model weights are excluded and must be downloaded separately under the upstream terms.

## Microsoft/Aurora-derived TC tracker

`agu_james/analysis/gnn/tc_tracker.py` and `agu_james/analysis/pangu/tracker_ver_aurora.py` preserve the original Microsoft MIT notice. The tracker docstring credits Anna Allen's original algorithm and Wessel Bruinsma's implementation/improvements. The upstream [Aurora MIT license](https://github.com/microsoft/aurora/blob/main/LICENSE.txt) is retained below; local adaptations are identified in provenance. Other author-written TC calculations are not automatically assigned this license.

MIT License

Implementation of the Aurora model.

Copyright (c) Microsoft Corporation.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## Data and dependencies

ERA5, JTWC, IBTrACS and WeatherBench data are external assets with their own terms/citations; see `docs/DATA_SOURCES.md`. Python/PyTorch/PyG/ONNX Runtime/Cartopy and other libraries are dependencies, not bundled source copies. Their installed versions/licenses should be recorded with the final environment. Original code is not relicensed by directory reorganization.
