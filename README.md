# Обучение модели

### 0. Environment

$\textbf{TODO}: \text{Add requirements file}$



```
conda activate fap2_env
```

Try not to install any additional libs to this env, if you need something, please notify me.


### 1. Data preparation

Install any data you want and add function of instalation to `.data/load.py`.

Specify the save path in `create_config.py`, you should fill `data.dataset_name`, `data.dataset_path`, then add the choice of a required function to main of `data/load.py`

Then run a file using following command
```
python -m data.load
```


### 2. Подсчет статистик датасета

Для обучения диффузионной модели необходимо предподсчитать покомпонентное среднее и дисперсию эмбеддингов текста диффузионного энкодеров.

Пример подсчета статистик для википедии для энкодера  [bert-base-uncased](https://huggingface.co/bert-base-uncased). представлен в файле `./data/make_encodings_mean_std.py`.

Если необходимо изменить энкодер, то нужно поменять в `config.py` поле `model.encoder_name` (название модели из huggingface)

Пример запуска:

```
python -m data.make_encodings_mean_std
```


### 3. Обучение декодера

BERT обучается на задачу демаскирования, поэтому, для того чтобы использовать его как автоэнкодер, необходимо дообучить на задачу реконструкции текста декодер.

Пример дообучения на википедии представлен в файле `./model/train_decoder.py`.

Пример запуска:
```
python -m model.train_decoder
```

Аналогично предыдущему пункту, если нужно заменить энкодер, необходимо заменить его в конфиге.

### 4. Обучение диффузионной модели

Обучение происходит в половинной точности и параллельно на нескольких видеокартах. 

Запуск обучения: `torchrun --nproc_per_node=4 --master_port=31345  train_diffuse_bert.py`

`nproc_per_node` -- количество видеокарт.





