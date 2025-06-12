# ViLaSR: Reinforcing Spatial Reasoning in Vision-Language Models with Interwoven Thinking and Visual Drawing

<!-- <div align="center"> -->
[![Paper](https://img.shields.io/badge/Paper-Arxiv-red)](https://arxiv.org/abs/2506.09965)
[![Paper](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-ViLaSR-blue)](https://huggingface.co/AntResearchNLP/ViLaSR)
<!-- [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT) -->
<!-- </div> -->


## 📢 News
- [2024.06] Data, code and model weights will be released soon. Please stay tuned! 🔥

## 📋 Overview
<p align="center">
    <img src="./assets/ViLaSR.png" width="90%" height="90%">
</p>

> As textual reasoning with large language models (LLMs) has advanced significantly, there has been growing interest in enhancing the multimodal reasoning capabilities of large vision-language models (LVLMs). However, existing methods primarily approach multimodal reasoning in a straightforward, text-centric manner, where both reasoning and answer derivation are conducted purely through text, with the only difference being the presence of multimodal input. As a result, these methods often encounter fundamental limitations in spatial reasoning tasks that demand precise geometric understanding and continuous spatial tracking—capabilities that humans achieve through mental visualization and manipulation. To address the limitations, we propose drawing to reason in space, a novel paradigm that enables LVLMs to reason through elementary drawing operations in the visual space. By equipping models with basic drawing operations, including annotating bounding boxes and drawing auxiliary lines, we empower them to express and analyze spatial relationships through direct visual manipulation, meanwhile avoiding the perfor- mance ceiling imposed by specialized perception tools in previous tool-integrated reasoning approaches. To cultivate this capability, we develop a three-stage train- ing framework: cold-start training with synthetic data to establish basic drawing abilities, reflective rejection sampling to enhance self-reflection behaviors, and re- inforcement learning to directly optimize for target rewards. Extensive experiments demonstrate that our model, named VILASR, consistently outperforms existing methods across diverse spatial reasoning benchmarks, involving maze navigation, static spatial reasoning, video-based reasoning, and multi-view-based reasoning tasks, with an average improvement of 18.4%. Ablation studies reveal the critical role of each training stage, where reflective rejection sampling strengthens the model’s self-correction capabilities, and reinforcement learning effectively unlocks its reasoning potential.

## 🚀 Coming Soon
- [ ] Model weights  
- [ ] Training code  
- [ ] Inference code  

## 📖 Citation
If you find our work helpful, please cite our paper:
```bibtex
@misc{wu2025reinforcingspatialreasoningvisionlanguage,
      title={Reinforcing Spatial Reasoning in Vision-Language Models with Interwoven Thinking and Visual Drawing}, 
      author={Junfei Wu and Jian Guan and Kaituo Feng and Qiang Liu and Shu Wu and Liang Wang and Wei Wu and Tieniu Tan},
      year={2025},
      eprint={2506.09965},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2506.09965}, 
}
```


## Acknowledgment
- [verl](https://github.com/volcengine/verl)
- [EasyR1](https://github.com/hiyouga/EasyR1)