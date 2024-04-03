# Multimodal DPAD with forensics trace disentanglement
## Overview
This is the implementation of the method proposed in "MULTI-MODAL DOCUMENT PRESENTATION ATTACK DETECTION WITH FORENSICS TRACE DISENTANGLEMENT" with tensorflow(1.12.0, gpu version).

## Introduction
This work proposes a DPAD method based on multi-modal disentangled traces (MMDT) without the above drawbacks. We first disentangle the recaptured traces by a self-supervised disentanglement and synthesis network to enhance the generalization capacity in document images with different contents and layouts. Then, unlike the existing DPAD approaches that rely only on data in the RGB domain, we propose to explicitly employ the disentangled recaptured traces as new modalities in the transformer backbone through adaptive multi-modal adapters to fuse RGB/trace features efficiently. Visualization of the disentangled traces confirms the effectiveness of the proposed method in different document contents. Extensive experiments on three benchmark datasets demonstrate the superiority of our MMDT method on representing forensic traces of recapturing distortion.
![Image text](https://github.com/chenlewis/)


## Reference
This work has been accepted by IEEE ICME2024.

[1] C. Chen, Y. Deng, L. Lin, Z. Yu, Z. Lai, "MULTI-MODAL DOCUMENT PRESENTATION ATTACK DETECTION WITH FORENSICS TRACE DISENTANGLEMENT," IEEE//2024 IEEE International Conference on Multimedia and Expo (ICME), Accepted Mar. 2024
