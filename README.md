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
baseline:
Col=0.08884052187204361, Dis=5.781296730041504, P_d_dist=594.4983520507812, V_loc=1.6174134016036987, A_loc=117.56974792480469, Energy=617.3888549804688, S_energy=884.631103515625, ADE = 3.0229239524473708, FDE = 5.872411841706393

baseline-flood:

baseline-kdma:


SFM:
Col=0.00800307933241129, Dis=0.9885196685791016, P_d_dist=413.21197509765625, V_loc=1.7698856592178345, A_loc=1.8902380466461182, Energy=245.6387176513672, S_energy=323.4501953125, ADE = 0.779524518245651, FDE = 1.1203974814640736

ORCA:
Col=0.0014795844908803701, Dis=0.9686275124549866, P_d_dist=412.7465515136719, V_loc=1.7209972143173218, A_loc=1.9885412454605103, Energy=228.5582275390625, S_energy=301.5916442871094, ADE = 0.7791208558398045, FDE = 1.0839182638942715
