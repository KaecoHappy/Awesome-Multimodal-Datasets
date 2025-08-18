# 🌐 Awesome-Multimodal-Datasets [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)

A curated list of high-quality multimodal datasets for AI research.  
> Modalities: Image, Video, 3D, Text, Audio, Sensor, etc.  

📣 **Always Updating** – new datasets & PRs welcome!  
If this repo helps you, **⭐ Star it** or **🤝 contribute** so the community keeps growing.

---

## 📚 Table of Contents
* [1 Training-Specific Datasets](#1-training-specific-datasets)  
  * [1.1 Image-Text](#11-image-text)  
  * [1.2 Video-Text](#12-video-text)  
  * [1.3 Audio-Text](#13-audio-text)  
  * [1.4 Audio-Video-Text](#14-audio-video-text)  
  * [1.5 3D-Text / Multi-Sensor](#15-3d-text--multi-sensor)  
  * [1.6 Instruction-Tuning](#16-instruction-tuning)  
* [2 Task-Specific Datasets](#2-task-specific-datasets)  
  * [2.1 Visual Question Answering (VQA)](#21-visual-question-answering-vqa)  
  * [2.2 Image & Video Captioning](#22-image--video-captioning)  
  * [2.3 Text-Video Retrieval](#23-text-video-retrieval)  
  * [2.4 Emotion & Behavior Recognition](#24-emotion--behavior-recognition)  
  * [2.5 Summarization & Retrieval](#25-summarization--retrieval)  
  * [2.6 Action Recognition & Localization](#26-action-recognition--localization)  
* [3 Domain-Specific Datasets](#3-domain-specific-datasets)  
  * [3.1 Medical & Healthcare](#31-medical--healthcare)  
  * [3.2 Autonomous Driving & Robotics](#32-autonomous-driving--robotics)  
  * [3.3 Geospatial & Remote Sensing](#33-geospatial--remote-sensing)  
  * [3.4 Industrial](#34-industrial)  
  * [3.5 Scientific & Scholarly](#35-scientific--scholarly)  
  * [3.6 Daily Activities / Egocentric Video](#36-daily-activities---egocentric-video)  
* [🌐 Other Resources](#-other-resources)  
* [🤝 Contributing](#-contributing)  
* [📝 License](#-license)  


---

## 1. Training-Specific Datasets  
> For **MM-PT** (pre-training) & **MM-IT** (instruction-tuning)

### 1.1 Image-Text
| Dataset | Modality | Size | Year | Label / Annotation | Notes | License |
|---------|----------|------|------|--------------------|-------| ------- |
| [MINT-1T](https://arxiv.org/abs/2406.11271) | Image-Text | 3.4B images, 1T tokens | 2024 | Image-Text pairs | includes PDFs, ArXiv papers |  |
| [BiomedCLIP](https://arxiv.org/abs/2303.00915) | Image-Text |15M biomedical image-text pairs | 2023 | Biomedical image-text pairs | Pretrained model for biomedical vision-language tasks |  |
| [Bloom Library](https://huggingface.co/datasets/bigscience/bloom-library) | Image-Text | 1.6 M pages | 2023 | Storybook text (many langs) | Low-resource languages; `multilingual` |
| [LAION-5B](https://laion.ai/blog/laion-5b) | Image-Text | 5.85 B pairs | 2022 | Alt-text + CLIP similarity + NSFW/watermark scores | includes alt-text, CLIP similarity, NSFW/watermark scores; 2.32 B English + 2.26 B multilingual; widely used for CLIP, Stable Diffusion; allowed training of many SOTA models. | CC-BY 4.0 |
| [COYO-700M](https://github.com/kakaobrain/coyo-dataset) | Image-Text | 747 M pairs | 2022 | Alt-text (web) | Common-Crawl, rich meta + pHash & NSFW scores|
| [LAION-COCO](https://laion.ai/blog/laion-coco) | Image-Text | 600 M pairs | 2022 | Synthetic COCO-style captions (BLIP+CLIP) | 600 M high-quality captions derived from LAION-5B English subset | CC-BY 4.0 |
| [Microsoft COCO](https://cocodataset.org/#home) | Image-Text | 330 K imgs | 2014 | Bounding boxes, instance masks, keypoints, category labels | Detection / segmentation |
| [Microsoft COCO Captions](https://cocodataset.org/#download) | Image-Text | 330 K imgs | 2014 | Human captions (5 or 40 sentences/img) | 5 human captions per image; widely used for captioning benchmark | CC-BY 4.0 |
| [Flickr30k](https://shannon.cs.illinois.edu/DenotationGraph) | Image-Text | 31 K imgs | 2014 |  Human captions (5 sentences per image) | 5 human captions per image; `real-world` scenes | CC-BY 2.0 |

### 1.2 Video-Text
| Dataset | Modality | Size | Year | Label / Annotation | Notes | License |
|---------|----------|------|------|--------------------|-------| ------- |
| [Video-MME](https://snap-research.github.io/Panda-70M) [[paper]](https://arxiv.org/abs/2402.19479) | Video-Text | Short (<2min), Medium (4–15min), Long (30–60min) videos | 2025 | Video-Text pairs | Comprehensive evaluation benchmark for multimodal LLMs in video analysis |
| [Panda-70M](https://snap-research.github.io/Panda-70M) [[paper]](https://arxiv.org/abs/2402.19479) | Video-Text | 70 M clips | 2024 | Multi-teacher captions + retrieval-filtered | 4100 hrs, cross-modal distilled descriptions; largest public high-res video-text corpus | 
| [OpenVid-1M](https://github.com/ali-vilab/OpenVid-1M) | Video-Text | 1 M clips | 2024 | High-quality captions | 720P; text-to-video generation | CC-BY 4.0 |
| [WebVid-10M](https://m-bain.github.io/webvid-dataset) | Video-Text | 10 M clips | 2021 | Web captions | 52,000 hrs; network-sourced clips with HTML alt-text | CC-BY 4.0 |
| [YouTube-8M](https://research.google.com/youtube8m) | Video-Text | 8 M videos | 2016 | 4716 class labels + auto captions | Large-scale retrieval, 5000 hrs; diverse categories; knowledge graph entities | CC-BY 4.0 |

### 1.3 Audio-Text
| Dataset | Modality | Size | Year | Label / Annotation | Notes | License |
|---------|----------|------|------|--------------------|-------|------- |
| [WavCaps](https://github.com/XinhaoMei/WavCaps) [[paper]](https://arxiv.org/abs/2303.17395) | Audio-Text | 400 k clips / 7600 hrs | 2023 | ChatGPT-filtered captions | Weak-label audio caption corpus; pre-training & captioning benchmark | CC-BY 4.0 |
| AudioCaps [[paper]](https://aclanthology.org/N19-1011) | Audio-Text | 46 K clips | 2019 | Human captions (1-5 per clip) | 10-s AudioSet clips; 128 hrs | CC-BY-NC 4.0 |
| [Clotho](https://zenodo.org/record/3490684) [[paper]](https://arxiv.org/abs/1910.09387) | Audio-Text | 4.9 K clips | 2020 | 5 human captions per clip | 15–30 s Freesound clips; DCASE audio-captioning benchmark | CC-BY 4.0 |
| [LibriTTS](https://openslr.org/60) [[paper]](https://arxiv.org/abs/1904.02882) | Audio-Text | 585 hrs | 2019 | Sentence-level transcripts + speaker metadata | 24 kHz, 2456 speakers; TTS & ASR benchmark |  CC-BY 4.0 |
| [AISHELL-2](https://www.aishelltech.com/aishell_2) | Audio-Text | 1000 hrs | 2018 | Manual transcripts | Mandarin ASR |
| [LJSpeech](https://keithito.com/LJ-Speech-Dataset) | Audio-Text | 13100 clips / 24 hrs | 2017 |  Public-domain transcripts | 22 kHz, single speaker; TTS & ASR benchmark | CC0 |

### 1.4 Audio-Video-Text
| Dataset | Modality | Size | Year | Label / Annotation | Notes | License |
|---------|----------|------|------|--------------------|-------| ------- |
| [InternVid](https://github.com/OpenGVLab/InternVid) [[paper]](https://arxiv.org/abs/2307.06942) | Video-Audio-Text | 234 M clips | 2023 | LLM-generated captions + ASR | 760 k hrs, 720P, 41 B words; largest LLM-annotated video-text corpus |
| [VGGSound](https://www.robots.ox.ac.uk/~vgg/data/vggsound/) | Video-Audio-Text | 200 K clips / 550 hrs | 2020 | 310-class audio labels | YouTube “in-the-wild” 10-s clips with **audio-visual correspondence**; train/test CSV provided | CC-BY-4.0 |

### 1.5 3D-Text / Multi-Sensor
| Dataset | Modality | Size | Year | Label / Annotation | Notes | License |
|---------|----------|------|------|--------------------|-------| ------- |
| [RSICap](https://shapenet.org) | Optical-SAR-Text | 104 k images | 2024 | Scene + object captions (color, shape, quantity, abs. location) |  |  |
| SoundingEarth | Aerial-Image-Audio-Text	 | 50 k image-audio pairs | 2023 | Aerial-Image-Audio-Text	50 k image-audio pairs | 5 crowd-sourced audio labels + scene descriptions | CC-BY 4.0 |
| [ShapeNet](https://shapenet.org) | 3D-Text | 51 K models | 2015 | Category labels + part hierarchy | CAD models; supports 3D classification, segmentation, and generation; widely used for pre-training 3D foundation models | CC-BY 4.0 |

### 1.6 Instruction-Tuning
| Dataset  | Modality   | Size  | Year | Label / Annotation  | Notes | License |
|----------|------------|-------|------|---------------------|-------| ------- |
| [LLaVA-Instruct-150K](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K) | Image-Text | ~150 K instruction pairs | 2023 | GPT-4 generated instructions| GPT-4–generated visual instructions based on COCO; three types: conversations, detailed, reasoning. Fine-tunes visual LLMs| CC-BY-4.0 |
| [InstructBLIP](https://github.com/salesforce/InstructBLIP)  | Image-Text | 1.6 M pairs | 2023 | Instruction-response pairs | Covers 11 tasks, 26 vision-language datasets into instruction-tuning format; introduces instruction-aware feature extraction; supports zero-shot evaluation and multi-modal tasks  | CC-BY 4.0 |
| [MIMIC-IT](https://github.com/Multimodal-Fatima/MIMIC-IT) | Image/Video-Text | 2.8 M pairs  | 2023 | Instruction-response pairs | 2.2 M unique instructions from images and videos; supports 8 languages; designed for zero-shot generalization and multi-modal context learning | CC-BY |
| [LAMM](https://opendatalab.com/LAMM/LAMM) [[paper]](https://arxiv.org/abs/2306.06687) | Image-Text, PointCloud-Text | 186 K image pairs, 10 K point cloud pairs | 2023 | Instruction-response pairs | 2D & 3D vision tasks; 3D reasoning tasks | AFL-3.0 |

---

## 2. Task-Specific Datasets  
> Evaluation benchmarks by task

### 2.1 Visual Question Answering (VQA)
| Dataset | Modality | Size | Year | Label / Annotation | Notes | License |
|---------|----------|------|------|--------------------|-------| ------- |
| [SlideVQA](https://slidevqa.github.io) | Image-Text |  14.5 K QA pairs | 2024 | QA pairs with bounding boxes and arithmetic expressions | 2.6k+ slide decks; requires complex reasoning including single-hop, multi-hop, and numerical reasoning | CC-BY 4.0 |
| [TVQA](https://tvqa.cs.unc.edu) | Video-Text | 152.5 K QA pairs | 2018 | QA pairs + timestamps | 21.8 K clips from 6 TV shows; compositional questions requiring joint localization, dialogue comprehension, and visual concept recognition | CC-BY 4.0 |
| [SQA3D](https://sqa3d.github.io) | Text-3D | 33.4 K questions | 2023  | QA pairs with situation descriptions | 6.8k unique situations from 650 ScanNet scenes; examines spatial reasoning, commonsense, navigation, and multi-hop logic; embodied AI QA |  CC-BY 4.0 |
| [MovieQA](http://movieqa.cs.toronto.edu) [[paper]](https://arxiv.org/abs/1512.02902) | Video-Text(EN) + audio | 14.9 K (14,944) QA pairs| 2016 | Multiple-choice QA pairs | 408 movies; questions require understanding of video clips, plots, subtitles, scripts, and DVS annotations; designed for evaluating story comprehension in movies | CC-BY 4.0 |

### 2.2 Image & Video Captioning
| Dataset | Modality | Size | Year | Label / Annotation | Notes | License |
|---------|----------|------|------|--------------------|-------| ------- |
| [MSR-VTT](https://www.microsoft.com/en-us/research/publication/msr-vtt-a-large-video-description-dataset-for-bridging-video-and-language) | Video-Text | 10 K clips / 200 K caps | 2016 | 20 human captions per clip | 10K web video clips; 20 captions each; 20 categories; supports video captioning, retrieval, and QA tasks |CC-BY 4.0|

### 2.3 Text-Video Retrieval
| Dataset | Modality | Size | Year | Label / Annotation | Notes | License |
|---------|----------|------|------|--------------------|-------| ------- |
| [WebVid-10M](https://m-bain.github.io/webvid-dataset) | Video-Text | 10 M clips | 2021 | Web captions | 52,000 hrs; network-sourced clips with HTML alt-text; widely used for text-video retrieval benchmark | CC-BY 4.0 |

### 2.4 Emotion & Behavior Recognition

| Dataset | Modality | Size | Year | Label / Annotation | Notes | License |
|---------|----------|------|------|--------------------|-------| ------- |
| [MAGB](https://github.com/sktsherlock/MAGB) | Audio-Text | Text-Image-Graph | 2019 | 7 emotions + 3 sentiment labels | 1,433 multi-speaker scenes from *Friends* TV show; 304 speakers; 3.6 s avg utterance; supports multimodal emotion recognition & dialogue sentiment analysis | CC-BY 4.0 |
| [MELD](https://affective-meld.github.io) | Audio-Text | 13,708 utterances | 2019 | 7 emotions + 3 sentiment labels | 1,433 multi-speaker scenes from *Friends* TV show; 304 speakers; 3.6 s avg utterance; supports multimodal emotion recognition & dialogue sentiment analysis | CC-BY 4.0 |
| [MuSe-CaR](https://sites.google.com/view/muse-2021/challenge/data) | Video-Audio-Text | 40 h 57 min | 2020 | 15-layer continuous emotion & trust labels | 70 hosts, 1.3 M frame-level labels; `real-world` YouTube car reviews; first large-scale driver trust & affect resource; supports embodied AI emotion analysis & customer-satisfaction automation|EULA for non-commercial use| 
| [IEMOCAP](https://sail.usc.edu/iemocap) | Video-Audio-Text | 12 hrs, 151 dialogs | 2008 | 10 emotions + dimensional labels | 10 actors; 7433 utterances; supports multimodal emotion recognition & dialogue sentiment analysis Dyadic acted | CC-BY 4.0 |
| [SEMAINE](https://semaine-db.eu) | Video-Audio-Text | 959 dialogs, 7.99 hrs | 2011 | Continuous Valence, Arousal, Expectancy, Power | 150 participants; 4 character personalities; supports multimodal emotion recognition & dialogue sentiment analysis; Sensitive listener | CC-BY 4.0 |
| [Belfast Emotional](http://www.psych.qub.ac.uk/BINED) [[paper]](https://ieeexplore.ieee.org/document/5480341) | Video-Audio | 750 clips (~327 min) | 2012 | Self-report intensity (0-9); continuous FeelTrace ratings of Valence & Intensity | Combination of Set 1 (570 clips, ~237 min) and Set 3 (180 clips, ~90 min); includes induction tasks (Frustration, Disgust, Fear, Surprise, Amusement); cross-cultural subset | CC-EULA |
| [MMAD (Micro-Action)]() | Video-Text | 6,528 videos, 19.8 k instances, 18.67 hrs | 2024 | 52 micro-action classes + temporal boundaries + multi-label co-occurrence | 	Cross-subject split; ACM MM 2025 Grand Challenge; subtle full-body movements | CC-BY 4.0 |
| [SAMSEMO](https://www.isca-archive.org/interspeech_2024/bujnowski24_interspeech.html) | Video-Audio-Text | 23 K video scenes in 5 languages | 2024 | Emotion categories & intensity | Multilingual multimodal emotion recognition dataset; supports cross-lingual emotion analysis | |
| [ViMACSA](https://arxiv.org/abs/2405.00543) |	Image-Text | 4,876 text-image pairs | 2024 | Aspect-category sentiment annotations | Vietnamese multimodal sentiment analysis dataset in hotel domain |  |

### 2.5 Summarization & Retrieval
| Dataset | Modality | Size | Year | Label / Annotation | Notes | License |
|---------|----------|------|------|--------------------|-------| ------- |
| [MAGB](https://github.com/sktsherlock/MAGB) | Graph-(Image+Text) | 5 graphs, 1.3 M nodes, 10 M edges | 2024 | Node classification & link-prediction labels; node-level image & text features | First multimodal attributed graph benchmark; 5 real-world e-commerce / book / fashion graphs; 4 pre-extracted feature sets provided | MIT |
| [MMSum](https://github.com/j-min/MMSum) | Video-Audio-Text | 5,100 videos, 170 subcategories, 1,229.9 hrs | 2023 | Human-validated video keyframes & text summaries (segment-level) | Multimodal summarization | CC BY-NC-SA 4.0 |

### 2.6 Action Recognition & Localization
> Action/step recognition and timing location in synthetic or real videos

| Dataset | Modality | Size  | Year | Label / Annotation | Notes | License |
| ------- | -------- | ----- | ---- | ------------------ | ----- | ------- |
| [DurLAR](https://arxiv.org/abs/2406.11271) | LiDAR-Image | 100,000 frames | 2024 | LiDAR and panoramic images | High-resolution 128-channel LiDAR data with panoramic images |
| [CATER](https://arxiv.org/abs/1910.04744) | Video |  5.5 k clips, 15.3 hrs, 10 s/clip | 2020 | Object attributes (shape/size/color), action labels with timestamps, snitch final position (6×6 grid)| 100 % `synthetic` tabletop  videos; 4 atomic actions (rotate, pick-place, slide, contain) composable into complex chains; zero human bias, ideal for diagnostic/`robotic` models |  |
| [HowTo100M](https://arxiv.org/abs/1906.03327) | Video-Text | 136 M clips / 1.22 M videos | 2019 | ASR narrated steps + verb-noun labels         | `real-world` Real YouTube instructional videos covering 23 k human tasks; 15-year total duration; perfect for learning real-world `human` action sequences |

---

## 3. Domain-Specific Datasets  
> Real-world verticals

### 3.1 Medical & Healthcare
| Dataset | Modality | Size | Year | Label / Annotation | Notes | License |
|---------|----------|------|------|--------------------|-------| ------- |
| [Medical](https://physionet.org/content/mimic-cxr-jpg/2.0.0) | Image-Text-Clinical Data | Data from 4,000 participants | 2025 | Diabetic eye research data | Focused on type 2 diabetes; standardized and multimodal |  |
| [MIMIC-CXR](https://physionet.org/content/mimic-cxr-jpg/2.0.0) | Image-Text | 377k chest X-ray images + 227k+ imaging studies + Structured metadata | 2019 | 14 standardized labels + Radiology reports | Covers 38 cardiopulmonary conditions; fully de-identified for privacy compliance; used in medical imaging detection, diagnosis, captioning | PhysioNet Credentialed Data Use Agreement |
| [PathGen-1.6M](https://github.com/PathLLM/PathGen) | Pathology images+text | 830k+ whole-slide histology images; 1.6M image-text pairs | 2024 | Text descriptions for representative image regions | Generated via multi-agent collaboration; supports AI-assisted pathological diagnosis |

### 3.2 Autonomous Driving & Robotics
| Dataset | Modality | Size | Year | Label / Annotation | Notes | License |
|---------|----------|------|------|--------------------|-------| ------- |
| [nuScenes](https://www.nuscenes.org) | Camera+LiDAR+Radar+GPS+IMU | 1 K scenes; ~1.4M camera images, 390K LiDAR sweeps, 1.4M radar scans；40,000 keyframes with 1.1B labeled LiDAR points | 2019 | 3D boxes, attributes, tracks | Urban driving; collected in Boston and Singapore; supports 3D object detection, tracking, and scene understanding| CC BY-SA 4.0 |
| [KITTI](http://www.cvlibs.net/datasets/kitti) | Grey&RGB Image+LiDAR+GPS/IMU | 6 hrs (25% publicly available) | 2012 | 3D bounding  boxes for 8 object categories | Designed for evaluating stereo vision, optical flow, visual odometry, 3D object detection, and 3D tracking in automotive scenarios; collected in Karlsruhe, Germany, covering urban, rural, and highway environments with diverse traffic conditions; data is calibrated, synchronized, and timestamped | CC BY-NC-SA 3.0 |
| [WayMoCo](http://arxiv.org/abs/2508.00589) | RGB Video + SMPL parameters + segmentation mask + text(motion & context) | 26 K seq | 2025 | SMPL pose + context text | An extension of Waymo Open Dataset providing 26,000 video sequences with SMPL motion parameters and context-aware text labels for multimodal human motion understanding in autonomous driving. | Apache-2.0 |
| [BASEPROD](https://www.nature.com/articles/s41597-024-03881-1) | RGB+Depth+Thermal+IMU | 2.4 K synthetic frames | 2024 | Pixel-wise 28-class segmentation + 3D boxes + navigability masks | RGB-D-T tri-modal robot perception benchmark covering real-world warehouse and corridor scenarios. |
| [DrivingDojo](https://github.com/phelber/eurosat) |Video-Sensor | ~18.2 K videos, 1920×1080, 5 FPS, total duration over 2 years | 2024 | Ego actions, agent interactions, rare open-world events | Large-scale driving video dataset from multiple Chinese cities; includes diverse weather and lighting conditions; interactive world models and Action Instruction Following (AIF) |  |

### 3.3 Geospatial & Remote Sensing
| Dataset | Modality | Size | Year | Label / Annotation | Notes | License |
|---------|----------|------|------|--------------------|-------| ------- |
| [SpaceNet](https://spacenet.ai) | Satellite imagery (electro-optical, SAR)-Text | 685,000+ building footprints; SpaceNet 6: 120+ sq km imagery, 48,000+ building labels; SpaceNet 7: 40,000+ km^2 imagery, 11M+ building footprints | Since 2016 | 	Building footprints, road networks, flood mapping labels, cross-modal registration labels; SpaceNet 7: unique building IDs for time tracking | Hosted on AWS; for geospatial ML, automated feature extraction; supports disaster response, urban analysis, environmental monitoring | CC BY-SA 4.0 |
| [EuroSAT](https://github.com/phelber/eurosat) | 	Satellite imagery + text | 27,000 images | 2019 | 10 land-cover classes | For land-use classification; covers European urban/rural areas; supports geospatial MLn | MIT |


### 3.4 Industrial
| Dataset | Modality | Size | Year | Label / Annotation | Notes | License |
|---------|----------|------|------|--------------------|-------| ------- |
| [MMAD(Industry)](https://arxiv.org/abs/2406.12345) | Image-Text | 8,366 industrial images; 39,672 multiple-choice questions | 2024 | Multiple-choice questions for 7 subtasks; Red-annotated anomaly regions; Prior knowledge of product categories and defect locations | Covers 38 product categories and 244 defect types; Aims to comprehensively evaluate MLLMs' industrial quality inspection capabilities; Multiple test settings to simulate real industrial scenarios | MIT |

### 3.5 Scientific & Scholarly
| Dataset | Modality | Size | Year | Label / Annotation | Notes | License |
|---------|----------|------|------|--------------------|-------| ------- |
| [Polymathic AI]() | Image-Text-Scientific data | 115 TB | 2024 |  Multidisciplinary scientific data | Astrophysics, biology, fluid dynamics, chemistry, acoustics | |
| [MMSci](https://github.com/Multimodal-Fatima/MMSci) | Image-Text | 200 K Q-V | 2024 |  Captions, scientific questions & answers (including multiple-choice) | From open-access scientific articles in Nature Communications; 72 scientific disciplines; for evaluating LMMs' scientific comprehension; used for visual instruction-following and pre-training | CC BY 4.0|
| [Multimodal ArXiv](https://aclanthology.org/2024.acl-long.775/) | Image-Text | 6.4 M images + 3.9 M captions | 2024 | Scientific figure captions & QA pairs | Multimodal scientific dataset for vision-language models; from 572K ArXiv papers | Large-scale multimodal daily activity dataset; first & third person views; 15 synchronized modalities |

### 3.6 Daily Activities / Egocentric Video
| Dataset | Modality | Size | Year | Label / Annotation | Notes | License |
|---------|----------|------|------|--------------------|-------| ------- |
|[CASTLE 2024](https://arxiv.org/abs/2503.17116)|Video-Audio-Sensor	|600+ hrs UHD video & multimodal data|	2024	Activity labels (egocentric & exocentric)	|Activity labels (egocentric & exocentric)||
## 🌐 Other Resources

* [**Papers with Code – Multimodal**](https://paperswithcode.com/task/multimodal-learning) - 
A comprehensive collection of benchmark tasks, datasets, and leaderboards for multimodal learning.

* [**MMBench**](https://github.com/pliang279/MultiBench) – All-in-One Multimodal Benchmark
NeurIPS 2021 one-stop benchmark & toolkit for multimodal model evaluation and development.
                          
---

## 🤝 Contributing

Found a cool dataset we missed? Want to add a new tag or category?  
Pull requests, issues, and suggestions are always welcome!

> Our starting taxonomy borrows from the framework in arXiv:2412.17759.  
> If the field moves faster than we do (it will!), help us keep this list fresh.

## 📝 License

This project is licensed under the [Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/), which allows for non-commercial use, sharing, and adaptation.

For commercial use, please contact the author for permission.
