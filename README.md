Training: `python train.py`
Evaluation: `python evaluate.py --LOAD_MODEL <MODEL>`
`<MODEL> = ./checkpoint/demonstration/model_final.bin`
Visualization: `python visualize.py --LOAD <MODEL>`

Install rvo2: https://juejin.cn/post/7297130301289332771

tensorboard --logdir <LOG_DIR>

Plot results of ppo
https://medium.com/aureliantactics/understanding-ppo-plots-in-tensorboard-cbc3199b9ba2
https://zhuanlan.zhihu.com/p/679975961

save env:
conda env export > environment.yml
conda list --explicit > spec-file.txt
pip freeze > requirements.txt

create env:
conda create --name ENV --file spec-file.txt
conda update--name ENV --file environment.yml
pip install -r requirements.txt

metrics:
tecrl 8.92, 7.77
sfm 0.88, 28.9
orca 1.93 11.2