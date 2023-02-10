# VK_MADE_DL_INTRO_HW

Решение задачи распознавания капчи с помощью RCNN модели. [Условие задания](https://docs.yandex.ru/docs/view?url=ya-disk-public%3A%2F%2FSsPMPReUk2c%2BHTFRqTkqgphiaj8CA0fdwfEBXz326JiXKBayrQeLvQERNOPE3Xcwq%2FJ6bpmRyOJonT3VoXnDag%3D%3D&name=laba.pdf).

Полное описание подхода описано в [description.ipynb](https://github.com/AmpiroMax/VK_MADE_DL_INTRO_HW/blob/master/description.ipynb)

Описание архитектуры проекта:

- В файле __dataset.py__ содержится класс `CaptchaDataset` считывающий изображения из папки `/data`
- В файле __model.py__ описан класс модели `RCNN`, почти полностью повторяющий архитектуру, предложенную в статье
- В файле __train.py__ описаны методы `train_epoch`, `eval_epoch` и `train`. Первые два проводят обучение модели и её валидацию на всех переданных им данных один раз. Метод `train` поочередно выполняет обучение и валидацию модели указанное число раз.
- В файле __pipeline.py__ описан метод `training_pipeline` запускающий обучения модели и вывод графиков Loss, CER. Также по итогу обучения, лучшая по CER модель сохраняетс в `/models/model`
- В Файле __analysis.py__ описан метод `searcher_for_problem_examples` поиска объектов на тестовой выборке, на которых модель допустила ошибку.
