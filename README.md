# Scene-aware Activity Program Generation with Language Guidance
Source codes of SIGGRAPH Asia 2023 Paper : "Scene-aware Activity Program Generation with Language Guidance"

![overview](./asset/teaser.png)

## Environment Setup

### Create a conda environment

We use two virtual environments in our project. The configuration of the virtual environment are:

- Python 3.11
- cuda 11.3
- Pytorch 1.13.0

You can create it using the following scripts:

```bash
conda create -n LVHomeEnv python=3.11
conda activate LVHomeEnv
git clone https://github.com/ToddBear/LangGuidedProgram
cd LangGuidedProgram
pip install -r requirements.txt
```

### Data Preparation

```bash
cd ..
mkdir Data
```

Download the program dataset [here](http://virtual-home.org), and organize it as follows:

```
Data
└── augment_programs
    ├── augment_exception
    └── augment_location
└── original_programs
```

Download the pre-computed future link [here](https://drive.google.com/file/d/1ibpqQrNTFep45ahEERhMd2L1m2H_YgYt/view?usp=sharing) for training the adjacency constraint moduel, and place it in Data folder:

```
Data
└── future_link.pkl
```

Fill in `LangGuidedProgram/program/dataset.py` line 28 with the absolute path to the parent folder `PARENT_PATH` of `LangGuidedProgram`.
```
PARENT_PATH = ""
program_path = PARENT_PATH + "/Data/original_programs/executable_programs/*/*/*txt"
augment_program_path = PARENT_PATH + "/Data/augment_programs/*/executable_programs/*/*/*/*txt"
sketch_path = PARENT_PATH + "/LangGuidedProgram/dataset/VirtualHome-Env/sketch_annotation.json"
```

## Training

Run the following scripts for training:

```bash
cd LangGuidedProgram/src
sh train.sh
```

## Testing

Download the pre-trained weights [here](https://drive.google.com/file/d/1JMEpH81oJaR3sYM5zcR08H_Gx3I6r88h/view?usp=drive_link), put it into the `LangGuidedProgram/src/output` folder, and run the following scripts:

```bash
cd LangGuidedProgram/src
sh test.sh
```

## Citation

If you find the code useful in your research, please consider citing:

```
@article{su2023scene,
  title={Scene-aware Activity Program Generation with Language Guidance Supplementary Material},
  author={Su, Zejia and Fan, Qingnan and Chen, Xuelin and Van Kaick, Oliver and Huang, Hui and Hu, Ruizhen},
  journal={ACM Trans. Graph},
  volume={42},
  number={6},
  year={2023}
}
```

## Acknowledgments

Our project is build upon [VirtualHome](https://github.com/xavierpuigf/virtualhome_unity) and [ResActGraph](https://github.com/andrewliao11/env-aware-program-gen). Thanks for your great constributions!

## License

Our code is released under MIT License. See LICENSE file for details.