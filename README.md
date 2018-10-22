# SST
Stochastic Substitute Training: A Gray-box Approach to Craft Adversarial Examples Against Gradient Obfuscation Defenses

This repository contains the code for Stochastic Substitute Training (SST) described in our paper to attack gradient obfuscation defenses in grey-box settings. We showed in 3 different Jupyter notebooks how to use SST to attack a non-robust model, a model fortified with Random Feature Nullification (RFN) and another model fortified with Thermometer Encoding. The code and hyper parameters used here are different from those we used for original experiments so the results are slightly different from those reported in the paper.

You need to install git-lfs (https://git-lfs.github.com/) to be able to download the pre-trained models.
After installing git-lfs you can just get this repo by using this command: 
```
git clone https://github.com/S-Mohammad-Hashemi/SST.git
```
Do not download the zip file from the GitHub's website directly. It doesn't download the pretrained-models.
## Paper

**Abstract:**

It has been shown that adversaries can craft example inputs to neural networks which are similar to legitimate inputs but have been created to purposely cause the neural network to misclassify the input. These adversarial examples are crafted, for example, by calculating gradients of a carefully defined loss function with respect to the input. As a countermeasure, some researchers have tried to design robust models by blocking or obfuscating gradients, even in white-box settings. Another line of research proposes introducing a separate detector to attempt to detect adversarial examples. This approach also makes use of gradient obfuscation techniques, for example, to prevent the adversary from trying to fool the detector. In this paper, we introduce stochastic substitute training, a gray-box approach that can craft adversarial examples for defenses which obfuscate gradients. For those defenses that have tried to make models more robust, with our technique, an adversary can craft adversarial examples with no knowledge of the defense. For defenses that attempt to detect the adversarial examples, with our technique, an adversary only needs very limited information about the defense to craft adversarial examples. We demonstrate our technique by applying it against two defenses which make models more robust and two defenses which detect adversarial examples.

## Citation

```
@inproceedings{SST,
 author = {Hashemi, Mohammad and Cusack, Greg and Keller, Eric},
 title = {Stochastic Substitute Training: A Gray-box Approach to Craft Adversarial Examples Against Gradient Obfuscation Defenses},
 booktitle = {Proceedings of the 11th ACM Workshop on Artificial Intelligence and Security},
 series = {AISec '18},
 year = {2018},
 isbn = {978-1-4503-6004-3},
 location = {Toronto, Canada},
 pages = {25--36},
 numpages = {12},
 url = {http://doi.acm.org/10.1145/3270101.3270111},
 doi = {10.1145/3270101.3270111},
 acmid = {3270111},
 publisher = {ACM},
 address = {New York, NY, USA},
} 

```
